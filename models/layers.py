import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import torch.nn.init as init

class HGNN_conv(nn.Module):
    def __init__(self, in_ft, out_ft, out_ft2, edge_num, bias=True):
        super(HGNN_conv, self).__init__()
        self.weight_location = Parameter(torch.Tensor(in_ft, out_ft))
        init.xavier_normal_(self.weight_location)
        self.weight_activity = Parameter(torch.Tensor(in_ft, out_ft))
        init.xavier_normal_(self.weight_activity)
        if bias:
            self.bias_location = Parameter(torch.Tensor(out_ft))
            self.bias_activity = Parameter(torch.Tensor(out_ft))
            init.zeros_(self.bias_location)
            init.zeros_(self.bias_activity)
        else:
            self.register_parameter('bias', None)
        self.fc = nn.Sequential(nn.Linear(out_ft, out_ft),nn.Linear(out_ft, out_ft2))
        self.act_leaky_relu = nn.LeakyReLU()

    def forward(self, x: torch.Tensor, left_location: torch.Tensor, right_location: torch.Tensor,\
                left_activity: torch.Tensor, right_activity: torch.Tensor):
        location_G = left_location.matmul(right_location)
        location_out1 = x.matmul(self.weight_location)
        if self.bias_location is not None:
            location_out1 = location_out1 + self.bias_location
        location_out2 = location_G.matmul(location_out1)
        location_out2 = self.act_leaky_relu(location_out2)

        activity_G = left_activity.matmul(right_activity)
        activity_out1 = x.matmul(self.weight_activity)
        if self.bias_activity is not None:
            activity_out1 = activity_out1 + self.bias_activity
        activity_out2 = activity_G.matmul(activity_out1)
        activity_out2 = self.act_leaky_relu(activity_out2)

        two_embed = location_out2 + activity_out2
        two_embed = two_embed/2
        final = self.fc(two_embed)
        return final

class HGNN_fc(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(HGNN_fc, self).__init__()
        self.fc = nn.Linear(in_ch, out_ch)

    def forward(self, x):
        return self.fc(x)

class HGNN_embedding(nn.Module):
    def __init__(self, in_ch, n_hid, dropout=0.5):
        super(HGNN_embedding, self).__init__()
        self.dropout = dropout
        self.hgc1 = HGNN_conv(in_ch, n_hid)
        self.hgc2 = HGNN_conv(n_hid, n_hid)

    def forward(self, x, G):
        x = F.relu(self.hgc1(x, G))
        x = F.dropout(x, self.dropout)
        x = F.relu(self.hgc2(x, G))
        return x

class HGNN_classifier(nn.Module):
    def __init__(self, n_hid, n_class):
        super(HGNN_classifier, self).__init__()
        self.fc1 = nn.Linear(n_hid, n_class)

    def forward(self, x):
        x = self.fc1(x)
        return x