import torch
import argparse

class Setting:
    def parse(self):
        self.guess_foursquare = True
        parser = argparse.ArgumentParser()
        self.parse_arguments(parser)                
        args = parser.parse_args(args=[])
        self.gpu = args.gpu
        self.hidden_dim = args.hidden_dim
        self.weight_decay = args.weight_decay
        self.learning_rate = args.lr
        self.epochs = args.epochs
        self.batch_size = args.batch_size
        self.dataset_file = './dataset/'+str(args.dataset_name1)+'/'+str(args.dataset_name)+'_'+str(args.region)+'_check_in.txt'
        self.user_embed_file = './dataset/'+str(args.dataset_name1)+'/user_embeddings.csv'
        self.user_sim_adj_file = './dataset/' + str(args.dataset_name1) + '/user_sim.csv'
        self.dataset_name1 = args.dataset_name1
        self.dataset_name = args.dataset_name
        self.region = args.region
        self.sequence_length = 20
        self.user_em_dim = args.user_em_dim
        self.loc_em_dim = args.loc_em_dim
        self.time_em_dim = args.time_em_dim
        self.motif_number = args.motif_number
        self.batch_size = args.batch_size
        self.max_users = 0
        self.min_checkins = 51
        self.lambda_t = args.lambda_t
        self.lambda_s = args.lambda_s
        self.walk_steps = args.walk_steps
        self.n_walks = args.n_walks
        self.validate_epoch = args.validate_epoch
        self.report_user = args.report_user
        self.device = torch.device('cpu') if args.gpu == -1 else torch.device('cuda', args.gpu)
    
    def parse_arguments(self, parser):        
        parser.add_argument('--gpu', default=0, type=int, help='the gpu to use')
        parser.add_argument('--dataset_name1', default='Foursquare_Yang', help='acceptable: Foursquare_Yang1')
        parser.add_argument('--dataset_name', default='Foursquare', help='acceptable: Foursquare_Yang')
        parser.add_argument('--region', default='NYC', help='acceptable: NYC, TKY')
        parser.add_argument('--hidden-dim', default=32, type=int, help='hidden dimensions to use')
        parser.add_argument('--user_em_dim', default=100, type=int, help='embed dimensions to use')
        parser.add_argument('--loc_em_dim', default=50, type=int, help='embed dimensions to use')
        parser.add_argument('--time_em_dim', default=10, type=int, help='embed dimensions to use')
        parser.add_argument('--motif_number', default=28, help='acceptable: 28')
        parser.add_argument('--n_walks', default=200, help='acceptable: 100')
        parser.add_argument('--walk_steps', default=10, help='acceptable: 5, 10')
        parser.add_argument('--weight_decay', default=0.0, type=float, help='weight decay regularization')
        parser.add_argument('--lr', default = 0.001, type=float, help='learning rate')
        parser.add_argument('--epochs', default=50, type=int, help='amount of epochs')
        parser.add_argument('--rnn', default='rnn', type=str, help='the GRU implementation to use: [rnn|gru|lstm]')
        parser.add_argument('--batch_size', default=64, type=int, help='amount of users to process in one pass (batching)') #16 64
        parser.add_argument('--lambda_t', default=0.1, type=float, help='decay factor for temporal data')
        parser.add_argument('--lambda_s', default=100, type=float, help='decay factor for spatial data')
        parser.add_argument('--dataset', default='foursquare_check_in.txt', type=str, help='the dataset under ./data/<dataset.txt> to load')
        parser.add_argument('--validate-epoch', default=5, type=int, help='run each validation after this amount of epochs')
        parser.add_argument('--report-user', default=-1, type=int, help='report every x user on evaluation (-1: ignore)')        

    def __str__(self):
        return ('parse with foursquare default settings' if self.guess_foursquare else 'parse with gowalla default settings') + '\n'\
            + 'use device: {}'.format(self.device)


        