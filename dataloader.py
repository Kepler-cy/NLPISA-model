import pandas as pd
from datetime import datetime

from dataset import PoiDataset, Usage

class PoiDataloader():
    def __init__(self, max_users=0, min_checkins=0,dataset_name=None):
        self.max_users = max_users
        self.min_checkins = min_checkins
        self.user2id = {}
        self.poi2id = {}
        self.activity_id = {}
        self.users = []
        self.times = []
        self.times_unix = []
        self.coords = []
        self.locs = []
        self.acts = []
        self.user_embedding = []
        self.user_2_user_sim = []
        self.dataset_name = dataset_name

    def create_dataset(self, sequence_length, batch_size, split, usage=Usage.MAX_SEQ_LENGTH, custom_seq_count=1):
        return PoiDataset(self.users.copy(),\
                          self.times.copy(),\
                          self.times_unix.copy(), \
                          self.coords.copy(),\
                          self.locs.copy(),\
                          self.acts.copy(), \
                          self.user_embedding.copy(), \
                          self.user_2_user_sim.copy(), \
                          sequence_length,\
                          batch_size,\
                          split,\
                          usage,\
                          len(self.poi2id), \
                          len(self.activity_id), \
                          custom_seq_count,self.dataset_name)

    def user_count(self):
        return len(self.users)
    
    def locations(self):
        return len(self.poi2id)

    def activities(self):
        return len(self.activity_id)

    def read(self, file, file_embed, file_sim):
        self.user_embedding = pd.read_csv(file_embed,index_col=0).values.tolist()
        self.user_2_user_sim = pd.read_csv(file_sim,index_col=0).values.tolist()
        for i in range(len(self.user_2_user_sim)):
            self.user_2_user_sim[i][i]=1.0
        self.read_users(file)
        self.read_pois(file)
    
    def read_users(self, file):        
        f = open(file, 'r')            
        lines = f.readlines()
        prev_user = int(lines[0].split('\t')[0])
        visit_cnt = 0
        for i, line in enumerate(lines):
            tokens = line.strip().split('\t')
            user = int(tokens[0])
            if user == prev_user and i!=len(lines)-1: # last user have no end!
                visit_cnt += 1
            else:
                if visit_cnt >= self.min_checkins:
                    self.user2id[prev_user] = len(self.user2id)
                else:
                   print('discard user {}: to few checkins ({})'.format(prev_user, visit_cnt))
                prev_user = user
                visit_cnt = 1
                if self.max_users > 0 and len(self.user2id) >= self.max_users:
                    break # restrict to max users

    def read_pois(self, file):
        f = open(file, 'r')
        lines = f.readlines()
        user_time = []
        user_time_unix = []
        user_coord = []
        user_loc = []
        user_act = []
        prev_user = int(lines[0].split('\t')[0])
        prev_user = self.user2id.get(prev_user)
        for i, line in enumerate(lines):
            tokens = line.strip().split('\t')
            user_old_id = int(tokens[0])
            if self.user2id.get(user_old_id) is None:
                continue
            user = self.user2id.get(user_old_id)
            time = int(tokens[7])
            is_workday = (datetime.weekday(datetime.strptime(tokens[1], "%Y-%m-%d %H:%M:%S+00:00"))+ 1) in range(1, 6)
            if is_workday==False: #consider weekdays and weekends
                time+=24
            time_unix = (datetime.strptime(tokens[1], "%Y-%m-%d %H:%M:%S+00:00") - datetime(1970, 1, 1)).total_seconds() # unix seconds
            lat = float(tokens[2])
            long = float(tokens[3])
            coord = (lat, long)
            location = int(tokens[4]) # location id
            activity = int(tokens[5]) # activity id
            if self.poi2id.get(location) is None:
                self.poi2id[location] = len(self.poi2id) #location id
            if self.activity_id.get(activity) is None: # get-or-set locations
                self.activity_id[activity] = len(self.activity_id) #get location id
            location = self.poi2id.get(location)
            activity = self.activity_id.get(activity)
            if user == prev_user:
                user_time.insert(0, time)
                user_time_unix.insert(0, time_unix)
                user_coord.insert(0, coord)
                user_loc.insert(0, location)
                user_act.insert(0, activity)
            else:
                if prev_user!=None:
                    self.users.append(prev_user)
                    self.times.append(user_time)
                    self.times_unix.append(user_time_unix)
                    self.coords.append(user_coord)
                    self.locs.append(user_loc)
                    self.acts.append(user_act)
                # resart:
                prev_user = user 
                user_time = [time]
                user_time_unix = [time_unix]
                user_coord = [coord]
                user_loc = [location]
                user_act = [activity]

        # process also the latest user in the for loop
        self.users.append(prev_user)
        self.times.append(user_time)
        self.times_unix.append(user_time_unix)
        self.coords.append(user_coord)
        self.locs.append(user_loc)
        self.acts.append(user_act)
