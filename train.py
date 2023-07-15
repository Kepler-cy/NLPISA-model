import datetime
import torch
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from setting import Setting
from trainer import NLPISATrainer
import utils.hypergraph_utils as hgut
from dataloader import PoiDataloader
from dataset import Split
from evaluation import Evaluation
import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
setting = Setting()
setting.parse()
poi_loader = PoiDataloader(setting.max_users, setting.min_checkins,setting.dataset_name1)
poi_loader.read(setting.dataset_file, setting.user_embed_file, setting.user_sim_adj_file)
dataset = poi_loader.create_dataset(setting.sequence_length, setting.batch_size, Split.TRAIN)
dataloader = DataLoader(dataset, batch_size=1, num_workers=1, shuffle=False)
dataset_test = poi_loader.create_dataset(setting.sequence_length, setting.batch_size, Split.TEST) #test data
dataloader_test = DataLoader(dataset_test, batch_size=1,num_workers=1, shuffle=False)
assert setting.batch_size < poi_loader.user_count(), 'batch size must be lower than the amount of available users'

#hgnn section
location_H = pd.read_csv('./dataset/' + str(setting.dataset_name1) + '/user_location_motif_num_nor.csv', index_col=0)
activcity_H = pd.read_csv('./dataset/' + str(setting.dataset_name1) + '/user_location_motif_num_nor.csv', index_col=0)
user2user_adj = pd.read_csv('./dataset/' + str(setting.dataset_name1) + '/user_2_user_sim_adj.csv', index_col=0)
user_features = pd.read_csv('./dataset/' + str(setting.dataset_name1) + '/user_feature.csv', index_col=0)
DV2_H_location_, _, invDE_HT_DV2_location_ = hgut._generate_G_from_H(location_H, variable_weight=True)  # Hypeedge matrix
DV2_H_activity_, _, invDE_HT_DV2_activity_ = hgut._generate_G_from_H(activcity_H, variable_weight=True)
# transform data to device
DV2_H_location = torch.Tensor(DV2_H_location_).to(setting.device)
invDE_HT_DV2_location = torch.Tensor(invDE_HT_DV2_location_).to(setting.device)
DV2_H_activity = torch.Tensor(DV2_H_activity_).to(setting.device)
invDE_HT_DV2_activity = torch.Tensor(invDE_HT_DV2_activity_).to(setting.device)
user_features = torch.Tensor(np.array(user_features)).to(setting.device)  # node features

# create NLPISATrainer trainer
trainer = NLPISATrainer(poi_loader.user_count(), poi_loader.locations(), poi_loader.activities(), setting.hidden_dim, setting.loc_em_dim,setting.time_em_dim,setting.user_em_dim,user_features.shape[1], setting.motif_number, user2user_adj, setting.walk_steps, setting.n_walks, setting.device)
trainer.prepare()
evaluation_test = Evaluation(dataset_test, dataloader_test, poi_loader.user_count(),trainer, setting)
optimizer = torch.optim.Adam(trainer.parameters(), lr=setting.learning_rate, weight_decay=setting.weight_decay)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20,30,40,60,80], gamma=0.5)

#%% model training
test_loss =[]
train_loss = []
s_all = []
min_loss = 9999
for e in range(setting.epochs):
    t1 = datetime.datetime.now()
    dataset.shuffle_users() # shuffle users before each epoch!
    losses = []
    for i, (s, t, t2, a, c, u, uht_l, uht_a, y, y_t, y_t2, y_a, y_c, act_locs, reset_h, active_users) in enumerate(dataloader):
        s = s.squeeze().to(setting.device)
        t = t.squeeze().to(setting.device)
        t2 = t2.squeeze().to(setting.device)
        a = a.squeeze().to(setting.device)
        c = c.squeeze().to(setting.device)
        u = u.squeeze().to(setting.device)
        uht_l = uht_l.squeeze().to(setting.device)
        uht_a = uht_a.squeeze().to(setting.device)
        y = y.squeeze().to(setting.device)
        y_a = y_a.squeeze().to(setting.device)
        active_users = active_users.to(setting.device)
        loss = trainer.loss(s, t, t2, a, c, u, uht_l, uht_a, y, y_a, act_locs,active_users, user_features, DV2_H_location,invDE_HT_DV2_location, DV2_H_activity, invDE_HT_DV2_activity)
        loss.backward(retain_graph=True)
        losses.append(loss.item())
        optimizer.step()
        optimizer.zero_grad()

    scheduler.step()
    # performance
    if (e+1) % 1 == 0:
        epoch_loss = np.mean(losses)
        print(f'Epoch: {e+1}/{setting.epochs}')
        print(f'Used learning rate: {scheduler.get_lr()[0]}')
        print(f'Avg Loss: {epoch_loss}')
        train_loss.append(epoch_loss)
        t4 = datetime.datetime.now()
        print("one epoch training time: ", (t4-t1).seconds)
    if (e+1) % 25  == 0:
        print(f'~~~ Test Set Evaluation (Epoch: {e+1}) ~~~')
        acc10,pre_results = evaluation_test.evaluate(user_features, DV2_H_location,invDE_HT_DV2_location, DV2_H_activity, invDE_HT_DV2_activity)
        test_loss.append(acc10)
        t5 = datetime.datetime.now()
        print("one epoch testing time: ", (t5 - t1).seconds)

torch.save(trainer.model.state_dict(), "./model_save/GRU_add_hgnn_social_activity.pth")