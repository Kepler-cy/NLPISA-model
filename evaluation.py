
import torch
import numpy as np

class Evaluation:
    
    '''
    Handles evaluation on a given POI dataset and loader.
    
    The two metrics are MAP and recall@n. Our model predicts sequencse of
    next locations determined by the sequence_length at one pass. During evaluation we
    treat each entry of the sequence as single prediction. One such prediction
    is the ranked list of all available locations and we can compute the two metrics.
    
    As a single prediction is of the size of all available locations,
    evaluation takes its time to compute. The code here is optimized.
    
    Using the --report_user argument one can access the statistics per user.
    '''
    
    def __init__(self, dataset, dataloader, user_count, trainer, setting):
        self.dataset = dataset
        self.dataloader = dataloader
        self.user_count = user_count
        self.trainer = trainer
        self.setting = setting

    def evaluate(self,user_features, DV2_H_location,invDE_HT_DV2_location, DV2_H_activity, invDE_HT_DV2_activity):
        self.dataset.reset()
        with torch.no_grad():        
            iter_cnt = 0
            recall1 = 0
            recall5 = 0
            recall10 = 0
            ndcg1 = 0
            ndcg5 = 0
            ndcg10 = 0
            average_precision = 0.
            real_and_pre_od = []
            u_iter_cnt = np.zeros(self.user_count)
            u_recall1 = np.zeros(self.user_count)
            u_recall5 = np.zeros(self.user_count)
            u_recall10 = np.zeros(self.user_count)
            u_ndcg1 = np.zeros(self.user_count)
            u_ndcg5 = np.zeros(self.user_count)
            u_ndcg10 = np.zeros(self.user_count)
            u_average_precision = np.zeros(self.user_count)        
            reset_count = torch.zeros(self.user_count)
            for i, (s, t, t2, a, c, u, uht_l, uht_a, y, y_t, y_t2, y_a, y_c, act_loc,reset_h, active_users) in enumerate(self.dataloader):
                active_users = active_users.squeeze()
                s = s.squeeze().to(self.setting.device)
                t = t.squeeze().to(self.setting.device)
                t2 = t2.squeeze().to(self.setting.device)
                a = a.squeeze().to(self.setting.device)
                c = c.squeeze().to(self.setting.device)
                u = u.squeeze().to(self.setting.device)
                uht_l = uht_l.squeeze().to(self.setting.device)
                uht_a = uht_a.squeeze().to(self.setting.device)
                y = y.squeeze()
                active_users = active_users.to(self.setting.device)
                # evaluate:
                out = self.trainer.evaluate(s, t, t2, a, c, u, uht_l, uht_a, act_loc,active_users,user_features, DV2_H_location,invDE_HT_DV2_location, DV2_H_activity, invDE_HT_DV2_activity) #[B,loc]
                for j in range(self.setting.batch_size):
                    ind_ = []
                    ind_.append(active_users[j].cpu().detach().numpy().reshape(-1)[:][0]);
                    ind_.append(s[-1, j].cpu().detach().numpy().reshape(-1)[:][0]);
                    ind_.append(y[j].cpu().detach().numpy().reshape(-1)[:][0]);
                    ind_.append(np.argmax(out[:, j].cpu().detach().numpy()));  # 存储real label
                    real_and_pre_od.append(ind_)

                    o = out[:,j] #time_steps * loc_num
                    y_j = y[j]
                    o_n = o.cpu().detach().numpy()
                    ind = np.argpartition(o_n, -10, axis=0)[-10:] # :, top 10 elements #在 -10:
                    # for k in range(len(y_j)):
                    if (reset_count[active_users[j]] > 1):
                        continue # skip already evaluated users.
                    # resort indices for k:
                    r = ind[np.argsort(-o_n[ind], axis=0)] # sort top 10 elements descending
                    r = torch.tensor(r)
                    t = y_j
                    r_kj = o_n[:]
                    t_val = r_kj[t]
                    upper = np.where(r_kj > t_val)[0]  #https://en.wikipedia.org/wiki/Mean_reciprocal_rank
                    precision = 1. / (1+len(upper))  #mean reciprocal rank
                    # store
                    u_iter_cnt[active_users[j]] += 1
                    u_recall1[active_users[j]] += t in r[:1]
                    u_ndcg1[active_users[j]] += (t in r[:1])
                    if t in r[:5]:
                        u_recall5[active_users[j]] += 1
                        rank_list = list(r[:5])
                        rank_index = rank_list.index(t)
                        u_ndcg5[active_users[j]] += 1.0 / np.log2(rank_index + 2)
                    if t in r[:10]:
                        u_recall10[active_users[j]] += t in r[:10]
                        rank_list = list(r[:10])
                        rank_index = rank_list.index(t)
                        u_ndcg10[active_users[j]] += 1.0 / np.log2(rank_index + 2)
                    u_average_precision[active_users[j]] += precision
            
            formatter = "{0:.8f}"
            for j in range(self.user_count):
                iter_cnt += u_iter_cnt[j]
                recall1 += u_recall1[j]
                recall5 += u_recall5[j]
                recall10 += u_recall10[j]
                ndcg1 += u_ndcg1[j]
                ndcg5 += u_ndcg5[j]
                ndcg10 += u_ndcg10[j]
                average_precision += u_average_precision[j]
                if (self.setting.report_user > 0 and (j+1) % self.setting.report_user == 0):
                    print('Report user', j, 'preds:', u_iter_cnt[j], 'recall@1', formatter.format(u_recall1[j]/u_iter_cnt[j]), 'MAP', formatter.format(u_average_precision[j]/u_iter_cnt[j]), sep='\t')
            print((recall1 / iter_cnt))
            print((recall5 / iter_cnt))
            print((recall10 / iter_cnt))
            print((ndcg1 / iter_cnt))
            print((ndcg5 / iter_cnt))
            print((ndcg10 / iter_cnt))
            print((average_precision/iter_cnt))
            print('Micro Recall@1:', formatter.format(recall1/iter_cnt))
            print('Micro Recall@5:', formatter.format(recall5/iter_cnt))
            print('Micro Recall@10:', formatter.format(recall10/iter_cnt))
            print('NDCG@1:', formatter.format(ndcg1/iter_cnt))
            print('NDCG@5:', formatter.format(ndcg5/iter_cnt))
            print('NDCG@10:', formatter.format(ndcg10/iter_cnt))
            print('MRR', formatter.format(average_precision/iter_cnt))
            print('predictions:', iter_cnt)

        return formatter.format(recall10/iter_cnt),real_and_pre_od