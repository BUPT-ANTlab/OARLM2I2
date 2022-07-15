import argparse
params = {"Epoch": 7200,                # 单次仿真最大步数
          "Episode": 10000,             # 总的训练轮数
          "test_episodes": 5,          # 测试时执行的轮数
          "lane_length": 100,           # sumo中每条车道的长度
          "use_global_reward": True,    # 是否使用全局的奖励
          "memory_capacity": 100000,    # 经验池的大小
          "warmup_phase": 1000,         # 热启动轮数
          "target_update_period": 400,  # 目标网络更新周期
          "gamma": 0.95,                # 奖励的折扣因子
          "alpha": 0.001,
          "assign_method": "task_allocation",    # 目标分配方式
          "train_pur": True,           # 设置训练目标：追逐车辆或者逃避车辆
          "evader_method": "Command",      # 逃避车辆运动方式
          "global_ob": True,            # 逃避车辆的观测
          # ==========================DQN=============================
          "epsilon_decay": 0.0001,      # 贪婪策略epsilon衰减
          "epsilon_min": 0.01,          # 贪婪策略epsilon最小值
          # ==========================MADDPG==========================
          "minimax": True
          }

parser = argparse.ArgumentParser(description='[Informer] Long Sequences Forecasting')

# ======================Reinforcement Setting=================================
parser.add_argument("--exp_name", type=str, default="test", help="addition name of the experiment")  # 实验名
parser.add_argument("--port", type=int, default=8813, help="The port of sumo environment")
parser.add_argument("--batch_size", type=int, default=2, help="The train batch size")
parser.add_argument("--domain_name", type=str, default="3x3Traffic", help="The domain of training")
parser.add_argument("--alg_name", type=str, default="PPO", help="The algorithm name of training")
parser.add_argument("--reload_exp", type=str, default=None, help="The reload exp name")
parser.add_argument("--test", action="store_true", default=False, help="Test")
parser.add_argument("--gui", action="store_true", default=False, help="Use gui")
parser.add_argument("--use_predict", action="store_true", default=False, help="Used to predict veh flow")
parser.add_argument("--pre_format", type=int, default=0, help="The format of predict to influence the time:"
                                                            " 0: pre[0] 1:pre[1] 2:pre[2] 3:pre[3] 4:pre[4]")
parser.add_argument("--model_name", type=str, default=None, help="The name of the model")
parser.add_argument("--assign_method", type=str, default="greedy", help="The method of assigning target")
parser.add_argument("--no_task", action="store_true", default=False, help="Don't use the task-assign")
parser.add_argument("--pre_method", type=str, default="informer", help="The method of predicting")
parser.add_argument("--ktr", action="store_true", default=False, help="Use Kronecker-Factored Trust Region to optimize the network")
parser.add_argument("--q_table_path", type=str, default=None, help="The q-table's path of evader")
parser.add_argument("--use_mau_loss", action="store_true", default=False, help="Use the mau loss to update model")
parser.add_argument("--use_encoder", action="store_true", default=False, help="Use the encoder")
parser.add_argument("--use_adj", action="store_true", default=False, help="Ues road adj to update")


# ======================Informer Setting ======================================
parser.add_argument('--model', type=str, required=False, default='informer',help='model of experiment, options: [informer, informerstack, informerlight(TBD)]')

parser.add_argument('--data', type=str, required=False, default='MVP1', help='data')
parser.add_argument('--root_path', type=str, default='./Informer-MVP/data/MVP/', help='root path of the data file')
parser.add_argument('--data_path', type=str, default='MVP3_3.csv', help='data file')
parser.add_argument('--features', type=str, default='M', help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
parser.add_argument('--freq', type=str, default='h', help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
parser.add_argument('--checkpoints', type=str, default='./Informer-MVP/checkpoints/', help='location of model checkpoints')

parser.add_argument('--seq_len', type=int, default=84, help='input sequence length of Informer encoder')
parser.add_argument('--label_len', type=int, default=42, help='start token length of Informer decoder')
parser.add_argument('--pred_len', type=int, default=28, help='prediction sequence length')
# Informer decoder input: concat[start token series(label_len), zero padding series(pred_len)]

parser.add_argument('--enc_in', type=int, default=48, help='encoder input size')
parser.add_argument('--dec_in', type=int, default=48, help='decoder input size')
parser.add_argument('--c_out', type=int, default=48, help='output size')

parser.add_argument('--d_model', type=int, default=128, help='dimension of model')
parser.add_argument('--n_heads', type=int, default=32, help='num of heads')
parser.add_argument('--e_layers', type=int, default=1, help='num of encoder layers')
parser.add_argument('--d_layers', type=int, default=2, help='num of decoder layers')
parser.add_argument('--s_layers', type=str, default='3,2,1', help='num of stack encoder layers')
parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')

parser.add_argument('--factor', type=int, default=5, help='probsparse attn factor')
parser.add_argument('--padding', type=int, default=0, help='padding type')
parser.add_argument('--distil', action='store_false', help='whether to use distilling in encoder, using this argument means not using distilling', default=True)
parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
parser.add_argument('--attn', type=str, default='prob', help='attention used in encoder, options:[prob, full]')
parser.add_argument('--embed', type=str, default='timeF', help='time features encoding, options:[timeF, fixed, learned]')
parser.add_argument('--activation', type=str, default='gelu',help='activation')
parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')
parser.add_argument('--mix', action='store_false', help='use mix attention in generative decoder', default=True)
parser.add_argument('--cols', type=str, nargs='+', help='certain cols from the data files as the input features')
parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
parser.add_argument('--itr', type=int, default=2, help='experiments times')
parser.add_argument('--train_epochs', type=int, default=500, help='train epochs')
# parser.add_argument('--batch_size', type=int, default=128, help='batch size of train input data')
parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=0.00001, help='optimizer learning rate')
parser.add_argument('--des', type=str, default='test',help='exp description')
parser.add_argument('--loss', type=str, default='mse',help='loss function')
parser.add_argument('--lradj', type=str, default='type1',help='adjust learning rate')
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)

parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
parser.add_argument('--devices', type=str, default='0,1,2,3',help='device ids of multile gpus')

args = parser.parse_args()



GLOBAL_SEED = 47
import torch
import numpy
import random
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.manual_seed(GLOBAL_SEED)
numpy.random.seed(GLOBAL_SEED)
random.seed(GLOBAL_SEED)
