from torch import nn
from models import HGNN_conv

class HGNN(nn.Module):
    def __init__(self, in_ch, edge_num, n_class, n_hid, dropout=0.5):
        super(HGNN, self).__init__()
        self.dropout = dropout
        self.edge_num = edge_num
        self.hgc1 = HGNN_conv(in_ch, n_hid, n_hid, edge_num)
        self.hgc2 = HGNN_conv(n_hid, n_hid, n_class, edge_num)

    def forward(self, x, left_location, right_location,left_activity, right_activity):
        layer1 = self.hgc1(x, left_location, right_location, left_activity, right_activity)
        layer2 = self.hgc2(layer1, left_location, right_location, left_activity, right_activity)
        return layer2
