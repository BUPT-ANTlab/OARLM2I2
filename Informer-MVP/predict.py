import argparse
import math

import numpy as np
import torch

import exp.exp_informer as EI
from utils.metrics import metric, MSE, RMSE
from settings import args

# parser = argparse.ArgumentParser(description='[Informer] Long Sequences Forecasting')
#
# # ======================Reinforcement Setting=================================
# parser.add_argument("--exp_name", type=str, default="test", help="addition name of the experiment")  # 实验名
# parser.add_argument("--port", type=int, default=8813, help="The port of sumo environment")
# parser.add_argument("--batch_size", type=int, default=32, help="The train batch size")
# parser.add_argument("--domain_name", type=str, default="3x3Traffic", help="The domain of training")
# parser.add_argument("--alg_name", type=str, default="PPO", help="The algorithm name of training")
# parser.add_argument("--reload_exp", type=str, default=None, help="The reload exp name")
# parser.add_argument("--test", action="store_true", default=False, help="Test")
# parser.add_argument("--gui", action="store_true", default=False, help="Use gui")
#
# # ======================Informer Setting ======================================
# parser.add_argument('--model', type=str, required=False, default='informer',help='model of experiment, options: [informer, informerstack, informerlight(TBD)]')
#
# parser.add_argument('--data', type=str, required=False, default='MVP1', help='data')
# parser.add_argument('--root_path', type=str, default='./Informer-MVP/data/MVP/', help='root path of the data file')
# parser.add_argument('--data_path', type=str, default='MVP3_3.csv', help='data file')
# parser.add_argument('--features', type=str, default='M', help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
# parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
# parser.add_argument('--freq', type=str, default='h', help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
# parser.add_argument('--checkpoints', type=str, default='./Informer-MVP/checkpoints/', help='location of model checkpoints')
#
# parser.add_argument('--seq_len', type=int, default=16, help='input sequence length of Informer encoder')
# parser.add_argument('--label_len', type=int, default=8, help='start token length of Informer decoder')
# parser.add_argument('--pred_len', type=int, default=5, help='prediction sequence length')
# # Informer decoder input: concat[start token series(label_len), zero padding series(pred_len)]
#
# parser.add_argument('--enc_in', type=int, default=48, help='encoder input size')
# parser.add_argument('--dec_in', type=int, default=48, help='decoder input size')
# parser.add_argument('--c_out', type=int, default=48, help='output size')
#
# parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
# parser.add_argument('--n_heads', type=int, default=50, help='num of heads')
# parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
# parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
# parser.add_argument('--s_layers', type=str, default='15,14,13,12,11,10,9,8,7,6,5,4,3,2,1', help='num of stack encoder layers')
# parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
#
# parser.add_argument('--factor', type=int, default=5, help='probsparse attn factor')
# parser.add_argument('--padding', type=int, default=0, help='padding type')
# parser.add_argument('--distil', action='store_false', help='whether to use distilling in encoder, using this argument means not using distilling', default=True)
# parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
# parser.add_argument('--attn', type=str, default='prob', help='attention used in encoder, options:[prob, full]')
# parser.add_argument('--embed', type=str, default='timeF', help='time features encoding, options:[timeF, fixed, learned]')
# parser.add_argument('--activation', type=str, default='gelu',help='activation')
# parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
# parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')
# parser.add_argument('--mix', action='store_false', help='use mix attention in generative decoder', default=True)
# parser.add_argument('--cols', type=str, nargs='+', help='certain cols from the data files as the input features')
# parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
# parser.add_argument('--itr', type=int, default=2, help='experiments times')
# parser.add_argument('--train_epochs', type=int, default=500, help='train epochs')
# # parser.add_argument('--batch_size', type=int, default=128, help='batch size of train input data')
# parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
# parser.add_argument('--learning_rate', type=float, default=0.00001, help='optimizer learning rate')
# parser.add_argument('--des', type=str, default='test',help='exp description')
# parser.add_argument('--loss', type=str, default='mse',help='loss function')
# parser.add_argument('--lradj', type=str, default='type1',help='adjust learning rate')
# parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
# parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)
#
# parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
# parser.add_argument('--gpu', type=int, default=0, help='gpu')
# parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
# parser.add_argument('--devices', type=str, default='0,1,2,3',help='device ids of multile gpus')
#
# args = parser.parse_args()

args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
#args.use_gpu =False

if args.use_gpu and args.use_multi_gpu:
    args.devices = args.devices.replace(' ','')
    device_ids = args.devices.split(',')
    args.device_ids = [int(id_) for id_ in device_ids]
    args.gpu = args.device_ids[0]

data_parser = {

'MVP1':{'data':'MVP3_3.csv','T':'OT','M':[48,48,48],'S':[1,1,1],'MS':[48,48,1]}
}
if args.data in data_parser.keys():
    data_info = data_parser[args.data]
    args.data_path = data_info['data']
    args.target = data_info['T']
    args.enc_in, args.dec_in, args.c_out = data_info[args.features]

args.s_layers = [int(s_l) for s_l in args.s_layers.replace(' ','').split(',')]
args.detail_freq = args.freq
args.freq = args.freq[-1:]

setting = '{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_at{}_fc{}_eb{}_dt{}_mx{}_{}_{}'.format(args.model, args.data, args.features,
                args.seq_len, args.label_len, args.pred_len,
                args.d_model, args.n_heads, args.e_layers, args.d_layers, args.d_ff, args.attn, args.factor,
                args.embed, args.distil, args.mix, args.des, 0)

class MVP_Informer():
    def __init__(self):
        self.storage_pool_predict=[]
        self.storage_pool_history = []
        self.exp=EI.Exp_Informer(args)
        #self.start=start_index
        #self.out_length = 10

    def pre(self):
        # t_current = t - self.start

        preds= self.exp.predict(setting,self.storage_pool_history,load=True)
        # self.store_pred(preds)
        # self.store_true(trues)

        return preds

    def save_his(self,his):
        self.storage_pool_history.append(his)







