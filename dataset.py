import copy
import random
from enum import Enum
import torch
from torch.utils.data import Dataset
import numpy as np

class Split(Enum):
    TRAIN = 0
    TEST = 1    

class Usage(Enum):
    '''
    Each user has a different amount of sequences. The usage defines
    how many sequences are used:
    MAX: each sequence of any user is used (default)
    MIN: only as many as the minimal user has
    CUSTOM: up to a fixed amount if available.
    The unused sequences are discarded. This setting applies after the train/test split.
    '''
    MIN_SEQ_LENGTH = 0
    MAX_SEQ_LENGTH = 1
    CUSTOM = 2

class PoiDataset(Dataset):
    '''
    Our Point-of-interest pytorch dataset: To maximize GPU workload we organize the data in batches of
    "user" x "a fixed length sequence of locations". The active users have at least one sequence in the batch.
    In order to fill the batch all the time we wrap around the available users: if an active user
    runs out of locations we replace him with a new one. When there are no unused users available
    we reuse already processed ones. This happens if a single user was way more active than the average user.
    The batch guarantees that each sequence of each user was processed at least once.
    
    This data management has the implication that some sequences might be processed twice (or more) per epoch.
    During trainig you should call PoiDataset::shuffle_users before the start of a new epoch. This
    leads to more stochastic as different sequences will be processed twice.
    During testing you *have to* keep track of the already processed users.    
    
    Working with a fixed sequence length omits awkward code by removing only few of the latest checkins per user.
    We work with a 80/20 train/test spilt, where test check-ins are strictly after training checkins.
    To obtain at least one test sequence with label we require any user to have at least (5*<sequence-length>+1) checkins in total.    
    '''
    
    def reset(self):
        # reset training state:
        self.next_user_idx = 0 # current user index to add
        self.active_users = [] # current active users
        self.active_user_seq = [] # current active users sequences
        self.user_permutation = [] # shuffle users during training
        
        # set active users:
        for i in range(self.batch_size):
            self.next_user_idx = (self.next_user_idx + 1) % len(self.users)
            self.active_users.append(i) 
            self.active_user_seq.append(0)
        
        # use 1:1 permutation:
        for i in range(len(self.users)):
            self.user_permutation.append(i)

    def shuffle_users(self):
        random.shuffle(self.user_permutation)
        # reset active users:
        self.next_user_idx = 0
        self.active_users = []
        self.active_user_seq = []
        for i in range(self.batch_size):
            self.next_user_idx = (self.next_user_idx + 1) % len(self.users)
            self.active_users.append(self.user_permutation[i]) 
            self.active_user_seq.append(0)
    
    def __init__(self, users, times,times_unix, coords, locs, acts, user_embed, user2user_adj, sequence_length, batch_size, split, usage, loc_count, act_count, custom_seq_count,dataset_name):
        self.users, self.times, self.times_unix, self.coords,self.locs,self.acts = users,times, times_unix, coords, locs, acts
        self.user_embed, self.user_2_user_adj = user_embed, user2user_adj
        self.split = split
        self.sequences = []
        self.activity_location = np.zeros((act_count,loc_count))
        self.locs_all = copy.deepcopy(locs)
        self.acts_all = copy.deepcopy(acts)
        self.sequences_times = []
        self.sequences_times_unix = []
        self.sequences_acts = []
        self.sequences_coords = []
        self.sequences_labels = []
        self.sequences_lbl_times = []
        self.sequences_lbl_times_unix = []
        self.sequences_lbl_acts = []
        self.sequences_lbl_coords = []
        self.sequences_count = []
        self.Ps = []
        self.Qs = torch.zeros(loc_count, 1)
        self.usage = usage
        self.batch_size = batch_size
        self.loc_count = loc_count
        self.act_count = act_count
        self.custom_seq_count = custom_seq_count
        self.user_loc_his_trans_pro = {}
        self.user_act_his_trans_pro = {}
        self.reset()
        self.dataset_name = dataset_name
        for i in range(loc_count): # location id and its index
            self.Qs[i, 0] = i

        #prepare for his adj extraction
        for i, (loc, act) in enumerate(zip(self.locs, self.acts)):
            for j in range(len(loc)):
                self.activity_location[act[j],loc[j]] = 1

        #his act and loc adj extraction
        for i, (loc1, act1) in enumerate(zip(self.locs_all, self.acts_all)):
            train_thr = int(len(loc1) * 0.8)
            self.locs_all[i] = loc1[:train_thr]
            self.acts_all[i] = act1[:train_thr]
        for i, (loc, act) in enumerate(zip(self.locs_all, self.acts_all)):
            user_ind = np.zeros((self.loc_count, self.loc_count)) #for social adj matrix
            user_ind2 = np.zeros((self.act_count, self.act_count)) #for social adj matrix
            for j in range(1,len(loc)):
                user_ind[loc[j-1], loc[j]]+=1
            for j in range(1,len(act)):
                user_ind2[act[j-1], act[j]]+=1
            self.user_loc_his_trans_pro[i] = user_ind
            self.user_act_his_trans_pro[i] = user_ind2

        # split to training / test phase:
        for i, (time,time_unix, act, coord, loc) in enumerate(zip(self.times,self.times_unix, self.acts, self.coords, self.locs)):
            train_thr = int(len(loc) * 0.8)
            if (split == Split.TRAIN):
                self.locs[i] = loc[:train_thr]
                self.times[i] = time[:train_thr]
                self.times_unix[i] = time_unix[:train_thr]
                self.acts[i] = act[:train_thr]
                self.coords[i] = coord[:train_thr]
            if (split == Split.TEST):
                self.locs[i] = loc[train_thr:]
                self.times[i] = time[train_thr:]
                self.times_unix[i] = time_unix[train_thr:]
                self.acts[i] = act[train_thr:]
                self.coords[i] = coord[train_thr:]

        # split location and labels to sequences:
        self.max_seq_count = 0
        self.min_seq_count = 10000000
        self.capacity = 0
        for i, (time, time_unix, act, coord, loc) in enumerate(zip(self.times, self.times_unix, self.acts, self.coords, self.locs)):
            seq_count = len(loc) - sequence_length
            seqs = []
            seq_times = []
            seq_times_unix = []
            seq_acts = []
            seq_coords = []
            seq_lbls = []
            seq_lbl_times = []
            seq_lbl_times_unix = []
            seq_lbl_acts = []
            seq_lbl_coords = []
            for j in range(seq_count):
                start = j
                end = j + sequence_length
                seqs.append(loc[start:end])
                seq_times.append(time[start:end])
                seq_times_unix.append(time_unix[start:end])
                seq_acts.append(act[start:end])
                seq_coords.append(coord[start:end])
                seq_lbls.append(loc[j+sequence_length])
                seq_lbl_acts.append(act[j+sequence_length])
                seq_lbl_times.append(time[j+sequence_length])
                seq_lbl_times_unix.append(time_unix[j+sequence_length])
                seq_lbl_coords.append(coord[j+sequence_length])
            self.sequences.append(seqs)
            self.sequences_times.append(seq_times)
            self.sequences_times_unix.append(seq_times_unix)
            self.sequences_acts.append(seq_acts)
            self.sequences_coords.append(seq_coords)            
            self.sequences_labels.append(seq_lbls)
            self.sequences_lbl_times.append(seq_lbl_times)
            self.sequences_lbl_times_unix.append(seq_lbl_times_unix)
            self.sequences_lbl_acts.append(seq_lbl_acts)
            self.sequences_lbl_coords.append(seq_lbl_coords)
            self.sequences_count.append(seq_count)
            self.capacity += seq_count
            self.max_seq_count = max(self.max_seq_count, seq_count)
            self.min_seq_count = min(self.min_seq_count, seq_count)

        # statistics
        if (self.usage == Usage.MIN_SEQ_LENGTH):
            print(split, 'load', len(users), 'users with min_seq_count', self.min_seq_count, 'batches:', self.__len__())
        if (self.usage == Usage.MAX_SEQ_LENGTH):
            print(split, 'load', len(users), 'users with max_seq_count', self.max_seq_count, 'batches:', self.__len__())
        if (self.usage == Usage.CUSTOM):
            print(split, 'load', len(users), 'users with custom_seq_count', self.custom_seq_count, 'Batches:', self.__len__())

    def sequences_by_user(self, idx):
        return self.sequences[idx]
    
    def __len__(self):
        if (self.usage == Usage.MIN_SEQ_LENGTH):
            return self.min_seq_count * (len(self.users) // self.batch_size)
        if (self.usage == Usage.MAX_SEQ_LENGTH):
            estimated = self.capacity // self.batch_size
            return max(self.max_seq_count, estimated)
        if (self.usage == Usage.CUSTOM):
            return self.custom_seq_count * (len(self.users) // self.batch_size)
        raise ValueError()
    
    def __getitem__(self, idx):
        ''' Against pytorch convention, we directly build a full batch inside __getitem__.
        Use a batch_size of 1 in your pytorch data loader.
        A batch consists of a list of active users,
        their next location sequence with timestamps and coordinates.
        y is the target location and y_t, y_s the targets timestamp and coordiantes. Provided for
        possible use.
        reset_h is a flag which indicates when a new user has been replacing a previous user in the
        batch. You should reset this users hidden state to initial value h_0.
        '''
        seqs = []
        times = []
        times_unix = []
        acts = []
        coords = []
        user_embeddings = []
        user_loc_his_trans_pro = []
        user_act_his_trans_pro = []
        lbls = []
        lbl_times = []
        lbl_times_unix = []
        lbl_acts = []
        lbl_coords = []
        reset_h = []
        for i in range(self.batch_size):
            i_user = self.active_users[i]
            j = self.active_user_seq[i]
            max_j = self.sequences_count[i_user]
            if (self.usage == Usage.MIN_SEQ_LENGTH):
                max_j = self.min_seq_count
            if (self.usage == Usage.CUSTOM):
                max_j = min(max_j, self.custom_seq_count) # use either the users maxima count or limit by custom count
            if (j >= max_j):
                # repalce this user in current sequence:
                i_user = self.user_permutation[self.next_user_idx]
                j = 0
                self.active_users[i] = i_user
                self.active_user_seq[i] = j
                self.next_user_idx = (self.next_user_idx + 1) % len(self.users)
                while self.user_permutation[self.next_user_idx] in self.active_users:
                    self.next_user_idx = (self.next_user_idx + 1) % len(self.users)
                # TODO: throw exception if wrapped around!
            # use this user:
            reset_h.append(j == 0)
            seqs.append(torch.tensor(self.sequences[i_user][j]))
            times.append(torch.tensor(self.sequences_times[i_user][j]))
            times_unix.append(torch.tensor(self.sequences_times_unix[i_user][j]))
            acts.append(torch.tensor(self.sequences_acts[i_user][j]))
            coords.append(torch.tensor(self.sequences_coords[i_user][j]))
            user_embeddings.append(torch.tensor(self.user_embed[i_user]))
            lbls.append(torch.tensor(self.sequences_labels[i_user][j]))
            lbl_times.append(torch.tensor(self.sequences_lbl_times[i_user][j]))
            lbl_times_unix.append(torch.tensor(self.sequences_lbl_times_unix[i_user][j]))
            lbl_acts.append(torch.tensor(self.sequences_lbl_acts[i_user][j]))
            lbl_coords.append(torch.tensor(self.sequences_lbl_coords[i_user][j]))
            current_loc_seq = self.sequences[i_user][j]
            current_act_seq = self.sequences_acts[i_user][j]
            user_ind = 0
            cadidate_loc_ind = (self.user_loc_his_trans_pro[user_ind][current_loc_seq[-1]]).copy()
            cadidate_act_ind = (self.user_act_his_trans_pro[user_ind][current_act_seq[-1]]).copy()
            if self.split == Split.TEST and j > 0 and user_ind == i_user:# update user his vis
                for i1 in range(j):
                    loc_seq = self.sequences[user_ind][i1]
                    act_seq = self.sequences_acts[user_ind][i1]
                    loc_mask = (loc_seq[:-1] == current_loc_seq[-1])
                    act_mask = (act_seq[:-1] == current_act_seq[-1])
                    cadidate_loc_ind[loc_seq[1:]] += loc_mask
                    cadidate_act_ind[act_seq[1:]] += act_mask
            # normlize cadidate_loc and added
            total_loc_ind = np.sum(cadidate_loc_ind)
            if total_loc_ind != 0:
                cadidate_loc_ind /= total_loc_ind
            total_act_ind = np.sum(cadidate_act_ind)
            if total_act_ind != 0:
                cadidate_act_ind /= total_act_ind
            cadidate_loc_ind = cadidate_loc_ind * self.user_2_user_adj[i_user][user_ind]
            cadidate_act_ind = cadidate_act_ind * self.user_2_user_adj[i_user][user_ind]
            for user_ind in range(1, len(self.sequences)):  # user num 417
                cadidate_loc = (self.user_loc_his_trans_pro[user_ind][current_loc_seq[-1]]).copy()
                cadidate_act = (self.user_act_his_trans_pro[user_ind][current_act_seq[-1]]).copy()
                if self.split == Split.TEST and j > 0 and user_ind==i_user:
                    for i1 in range(j):
                        loc_seq = self.sequences[user_ind][i1]
                        act_seq = self.sequences_acts[user_ind][i1]
                        loc_mask = (loc_seq[:-1] == current_loc_seq[-1])
                        act_mask = (act_seq[:-1] == current_act_seq[-1])
                        cadidate_loc[loc_seq[1:]] += loc_mask
                        cadidate_act[act_seq[1:]] += act_mask
                total_loc = np.sum(cadidate_loc)
                total_act = np.sum(cadidate_act)
                if total_loc != 0:
                    cadidate_loc /= total_loc
                if total_act != 0:
                    cadidate_act /= total_act
                cadidate_loc = cadidate_loc*self.user_2_user_adj[i_user][user_ind]
                cadidate_act = cadidate_act*self.user_2_user_adj[i_user][user_ind]
                cadidate_loc_ind = cadidate_loc_ind + cadidate_loc
                cadidate_act_ind = cadidate_act_ind + cadidate_act
            user_loc_his_trans_pro.append(torch.tensor(np.array(cadidate_loc_ind)))
            user_act_his_trans_pro.append(torch.tensor(np.array(cadidate_act_ind)))
            self.active_user_seq[i] += 1
        s = torch.stack(seqs, dim=1)
        t = torch.stack(times, dim=1)
        t2 = torch.stack(times_unix, dim=1)
        a = torch.stack(acts, dim=1)
        c = torch.stack(coords, dim=1)
        u = torch.stack(user_embeddings, dim=1)
        uht_l = torch.stack(user_loc_his_trans_pro, dim=1)  #[user,B,loc]
        uht_a = torch.stack(user_act_his_trans_pro, dim=1)  #[user,B,loc]
        y = torch.stack(lbls, dim=-1)
        y_t = torch.stack(lbl_times, dim=-1)
        y_t2 = torch.stack(lbl_times_unix, dim=-1)
        y_a = torch.stack(lbl_acts, dim=-1)
        y_c = torch.stack(lbl_coords, dim=-1)
        return s, t, t2, a, c, u, uht_l, uht_a, y, y_t, y_t2, y_a, y_c, torch.tensor(self.activity_location), reset_h, torch.tensor(self.active_users)#reset_h,sa,

