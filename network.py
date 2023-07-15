import torch
import torch.nn as nn
from models import HGNN_conv

class NLPISA(nn.Module):
    def __init__(self, user_count, location_count, act_count, hidden_size, loc_em_dim, time_em_dim, user_em_dim, in_ch, edge_num):
        super().__init__()
        self.user_count = user_count #417
        self.location_num = location_count #4045
        self.activity_num = act_count #417
        self.inch = in_ch
        self.edge_num = edge_num

        self.loc_emb = nn.Embedding(self.location_num, loc_em_dim) # location embedding
        self.act_emb = nn.Embedding(self.activity_num, loc_em_dim) # activity embedding
        self.time_emb = nn.Embedding(48, time_em_dim) # user embedding
        self.user_emb = nn.Embedding(self.user_count, user_em_dim) # user embedding

        self.rnn_a = nn.GRU(loc_em_dim + time_em_dim,  hidden_size) # [120, 64]
        self.rnn_s = nn.GRU(loc_em_dim + time_em_dim, hidden_size) # [120, 64]
        self.init_weights()

        self.fc_a = nn.Linear(hidden_size + user_em_dim, self.activity_num) # create outputs in lenght of locations
        self.fc_s = nn.Linear(hidden_size + user_em_dim + hidden_size, self.location_num) # create outputs in lenght of locations

        #not use
        self.hgc1 = HGNN_conv(self.inch, hidden_size*2, hidden_size*2, self.edge_num)
        self.hgc2 = HGNN_conv(hidden_size*2, hidden_size*2, user_em_dim, edge_num)

        self.loc_gate_a = nn.Linear(self.activity_num,1)
        self.loc_gate_s = nn.Linear(self.location_num,1)

        self.softmax = nn.Softmax(dim=1)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(0.5)

    def init_weights(self):
        ih = (param.data for name, param in self.named_parameters() if 'weight_ih' in name)
        hh = (param.data for name, param in self.named_parameters() if 'weight_hh' in name)
        b = (param.data for name, param in self.named_parameters() if 'bias' in name)
        for t in ih:
            nn.init.xavier_uniform_(t)
        for t in hh:
            nn.init.orthogonal_(t)
        for t in b:
            nn.init.constant_(t, 0)

    def forward(self, s, t, t2, a, c, u, uht_l, uht_a, act_loc,active_users, user_features, left_loc, right_loc, left_act, right_act, device): #seqs time activity, corrds, user  sa,
        s_emb = self.loc_emb(s) #[steps,B,hidden]
        t_emb = self.time_emb(t) #[steps,B,hidden]
        a_emb = self.act_emb(a) #[steps,B,hidden]
        u_emb = self.user_emb(active_users)
        u= u.transpose(0, 1)
        user_embed_mul = torch.mul(u_emb, u)

        '''*****************Task 1: activity prediction*************'''
        emb_cate_a = torch.cat((a_emb, t_emb), dim=-1)
        emb_cate_a = self.dropout(emb_cate_a)
        out_rnn_main_a, h_a = self.rnn_a(emb_cate_a)
        out_rnn_main_a_add = torch.cat((out_rnn_main_a[-1,:,:], user_embed_mul.squeeze(0)), dim=-1) # hgnn association
        y_linear_a = self.fc_a(out_rnn_main_a_add)
        uht_a = uht_a.transpose(0, 1)  # [B,act]
        y_linear_a_gate = self.tanh(self.loc_gate_a(y_linear_a))
        left_1 = torch.mul(y_linear_a_gate, y_linear_a)  # save old
        right_1 = torch.mul(1 - y_linear_a_gate, uht_a)  # consider social
        y_linear_a_with_social = torch.add(left_1, right_1)
        activity_Cons = torch.matmul(y_linear_a_with_social.double(), act_loc.to(device).double()).squeeze(0)

        '''*****************Task 2: location prediction*************'''
        emb_cate_s = torch.cat((s_emb, t_emb), dim=-1)
        emb_cate_s = self.dropout(emb_cate_s)
        out_rnn_main_s, h_s = self.rnn_s(emb_cate_s)
        out_rnn_main_s_add = torch.cat((out_rnn_main_s[-1,:,:], user_embed_mul.squeeze(0)), dim=-1)
        out_rnn_main_s_add = torch.cat((out_rnn_main_s_add, out_rnn_main_a[-1, :, :]), dim=-1)
        out_rnn_main_s_add = self.dropout(out_rnn_main_s_add)
        y_linear_s = self.fc_s(out_rnn_main_s_add)
        uht_l = uht_l.transpose(0, 1)  #[B,loc]
        y_linear_s_gate = self.tanh(self.loc_gate_s(y_linear_s))
        left_2 = torch.mul(y_linear_s_gate, y_linear_s)
        right_2 = torch.mul(1-y_linear_s_gate, uht_l)
        y_linear_s_with_social = torch.add(left_2, right_2)
        # Activity association2
        y_linear_add_hgnn_social_activity = torch.add(y_linear_s_with_social, activity_Cons)
        return y_linear_add_hgnn_social_activity, y_linear_a_with_social
