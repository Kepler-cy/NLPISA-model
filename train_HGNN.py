import os
import time
import torch
from torch import nn
import argparse
import numpy as np
import pandas as pd
from models import HGNN
from models import RandomWalker
import torch.optim as optim
import utils.hypergraph_utils as hgut

parser = argparse.ArgumentParser()
parser.add_argument('--gpu_id', default='0', help='Visible GPU id')
parser.add_argument('--model_version', default='HGCN', help='HGCN model version')
parser.add_argument('--dataset_name1', default='Foursquare_Yang', help='acceptable: Foursquare_Yang1')
parser.add_argument('--dataset_name', default='Foursquare', help='acceptable: Foursquare_Yang')
parser.add_argument('--region', default='TKY', help='acceptable: NYC, TKY,Geolife')
parser.add_argument('--user_number', default=1208, help='acceptable: 417,1208')
parser.add_argument('--location_number', default=6936, help='acceptable: 4045,6936')
parser.add_argument('--activity_number', default=282, help='acceptable: 302,282')
parser.add_argument('--motif_number', default=28, help='acceptable: 28')
parser.add_argument('--n_layers', default=2, help='acceptable: 2,3,4')
parser.add_argument('--has_bias', default=True, help='acceptable: True')
parser.add_argument('--add_self_loop', default=True, help='acceptable: True')
parser.add_argument('--n_walks', default=200, help='acceptable: 200')
parser.add_argument('--walk_steps', default=10, help='acceptable: 5, 10')
parser.add_argument('--hidden_num', default=16, help='acceptable: 16,32')
parser.add_argument('--embedding_dim', default=100, help='acceptable: 200,100')
parser.add_argument('--lr', default=0.0001, help='acceptable: 0.0001')
parser.add_argument('--weight_decay', default=0.0001, help='acceptable: 0.005')
parser.add_argument('--drop_out', default=0.3, help='acceptable: 0.001')
parser.add_argument('--print_freq', default=10, help='acceptable: 500,100')
parser.add_argument('--max_epoch', default=200, help='acceptable: 500,200,100')
args = parser.parse_args(args=[])
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
log_sig = nn.LogSigmoid()

location_H = pd.read_csv('./dataset/' + str(args.dataset_name1) + '/user_location_motif_num.csv', index_col=0)
activcity_H = pd.read_csv('./dataset/' + str(args.dataset_name1) + '/user_location_motif_num.csv', index_col=0)
user2user_adj = pd.read_csv('./dataset/' + str(args.dataset_name1) + '/user_2_user_sim_adj.csv', index_col=0)
user_features = pd.read_csv('./dataset/' + str(args.dataset_name1) + '/user_feature.csv', index_col=0)
DV2_H_location_, _, invDE_HT_DV2_location_ = hgut._generate_G_from_H(location_H, variable_weight=True)  # H-edge matrix
DV2_H_activity_, _, invDE_HT_DV2_activity_ = hgut._generate_G_from_H(activcity_H, variable_weight=True)  # H-edge matrix
DV2_H_location = torch.Tensor(DV2_H_location_).to(device)
invDE_HT_DV2_location = torch.Tensor(invDE_HT_DV2_location_).to(device)
DV2_H_activity = torch.Tensor(DV2_H_activity_).to(device)
invDE_HT_DV2_activity = torch.Tensor(invDE_HT_DV2_activity_).to(device)
user_features = torch.Tensor(np.array(user_features)).to(device)

def loss_function(embed_results):
    walker = RandomWalker.RW(user2user_adj.values, rw_len = args.walk_steps, n_walks = args.n_walks)
    walk_seq = walker.walk().__next__()
    similarity = 0.0
    for i in range(len(walk_seq)):
        for j in range(1, len(walk_seq[i])):
            a = embed_results[walk_seq[i][j-1], :]
            b = embed_results[walk_seq[i][j], :]
            in_ = torch.tensor(np.inner(a, b), device=device.type)
            similarity -= log_sig(in_)
    similarity /= (args.walk_steps * args.n_walks)
    return similarity

def train_model(model, optimizer, num_epochs, print_freq):
    since = time.time()
    for epoch in range(num_epochs):
        if epoch % print_freq == 0:
            print('-' * 10)
            print(f'Epoch {epoch}/{num_epochs}')
        model.train()
        running_loss = 0.0
        optimizer.zero_grad()
        embedding_results = model(user_features, DV2_H_location, invDE_HT_DV2_location, DV2_H_activity, invDE_HT_DV2_activity)
        loss = loss_function(embedding_results.detach().cpu().numpy())
        loss.requires_grad_(True)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if epoch % print_freq == 0:
            print(f'Training Loss: {running_loss:.4f}')
    time_elapsed = time.time() - since
    print(f'\nTraining complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    return model

def _main():
    model_hgnn = HGNN(in_ch = user_features.shape[1],
                    edge_num = args.motif_number,
                    n_class = args.embedding_dim,
                    n_hid = args.hidden_num,
                    dropout = args.drop_out)
    model_hgnn = model_hgnn.to(device)
    optimizer = optim.Adam(model_hgnn.parameters(), lr=args.lr)
    model_hgnn = train_model(model_hgnn, optimizer, args.max_epoch, print_freq=args.print_freq)
    embedding_results_final = model_hgnn(user_features, DV2_H_location, invDE_HT_DV2_location, DV2_H_activity, invDE_HT_DV2_activity).detach().cpu().numpy()
    return embedding_results_final, model_hgnn

if __name__ == '__main__':
    re, model_ = _main()