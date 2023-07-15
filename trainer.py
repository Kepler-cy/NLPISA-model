import torch.nn as nn
from network import NLPISA

class NLPISATrainer():
    def __init__(self, user_count, loc_count, act_count, hidden_size, loc_em_dim, time_em_dim, user_em_dim, in_ch, edge_num, user2user_adj, walk_steps, n_walks, device):
        self.user_count = user_count
        self.loc_count = loc_count
        self.act_count = act_count
        self.hidden_size = hidden_size
        self.loc_em_dim = loc_em_dim
        self.time_em_dim = time_em_dim
        self.user_em_dim = user_em_dim
        self.device = device
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.inch = in_ch
        self.edge_num = edge_num
        self.user2user_adj = user2user_adj
        self.walk_steps = walk_steps
        self.n_walks = n_walks

    def prepare(self):
        self.model = NLPISA(self.user_count, self.loc_count, self.act_count, self.hidden_size, \
                            self.loc_em_dim, self.time_em_dim, self.user_em_dim, self.inch, self.edge_num).to(self.device)

    def __str__(self):
        return 'Use our NLPISA training.'

    def parameters(self):
        return self.model.parameters()

    def evaluate(self, s, t, t2, a, c, u, uht_l, uht_a, act_loc,active_users,user_features, left_loc, right_loc, left_act,right_act):
        self.model.eval()
        out, out_act = self.model(s, t, t2, a, c, u, uht_l, uht_a, act_loc,active_users, user_features, left_loc, right_loc, left_act,right_act, self.device) #[B,loc]
        out_t = out.transpose(0, 1)
        return out_t

    # loss FUN
    def loss(self, s, t, t2, a, c, u, uht_l, uht_a, y, y_a, act_loc, active_users, user_features, left_loc, right_loc, left_act,right_act):
        self.model.train()
        out, out2 = self.model(s, t, t2, a, c, u, uht_l, uht_a, act_loc, active_users, user_features, left_loc, right_loc, left_act,right_act, self.device)
        out = out.view(-1, self.loc_count)
        y = y.view(-1)
        l = self.cross_entropy_loss(out, y)
        out2 = out2.view(-1, self.act_count)
        y_a = y_a.view(-1)
        l2 = self.cross_entropy_loss(out2, y_a)
        return l + l2

