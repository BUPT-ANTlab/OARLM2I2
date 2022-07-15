from agent.controller import DeepLearningController, ReplayMemory
from agent.modules import MLP, PursuitModule, UPDeT, MLPencoder, EncOut
from agent.utils import argmax, get_param_or_default
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy
from agent.kfac import KFACOptimizer
from torch.distributions import Categorical
from sklearn.metrics import mutual_info_score

class DQNNet(torch.nn.Module):
    def __init__(self, input_shape, outputs, max_history_length, params=None):
        super(DQNNet, self).__init__()
        self.params = params
        if params["use_encoder"]:
            self.enc_out_net = EncOut((1, 132))
            self.encoder_net = MLPencoder(input_shape, max_history_length, params=params)
        self.fc_net = MLP(input_shape, max_history_length, params=params)
        self.action_head = nn.Linear(self.fc_net.nr_hidden_units, outputs)


    def forward(self, x):# 4*4*33
        if x.ndim == 6:
            batch_size = x.size(0)
            num_agent = x.size(2)
            x = x.view(batch_size * num_agent, x.size(3), x.size(4), x.size(5))
            if self.params["use_encoder"]:
                self.encoder_output = self.encoder_net(x[:, 0:3])
                current_x = x[:, -1]
                current_x = current_x.view(x.size(0), -1)
                self.fc_output = self.enc_out_net(current_x)
                x = torch.cat([current_x, self.encoder_output], 1)
                x = self.fc_net(x)
                return F.softmax(self.action_head(x), dim=-1)
            else:
                current_x = x[:, -1]
                current_x = current_x.view(x.size(0), -1)
                x = current_x
                x = self.fc_net(x)
                return F.softmax(self.action_head(x), dim=-1)
        else:
            if self.params["use_encoder"]:
                self.encoder_output = self.encoder_net(x[:, 0:3])  # 取前三个输入；输出是1*48
                current_x = x[:, -1]
                current_x = current_x.view(x.size(0), -1)  # 1*(4*33)
                self.fc_output = self.enc_out_net(current_x)  # 输入一个全连接层，得到1*48
                x = torch.cat([current_x, self.encoder_output], 1)
                x = self.fc_net(x)
                return F.softmax(self.action_head(x), dim=-1)
            else:
                current_x = x[:, -1]
                current_x = current_x.view(x.size(0), -1)  # 1*(4*33)
                x = current_x
                x = self.fc_net(x)
                return F.softmax(self.action_head(x), dim=-1)


class DQNUPDeT(nn.Module):
    def __init__(self, input_shape, nr_actions, max_history_length, device="cuda", params=None):
        super(DQNUPDeT, self).__init__()
        self.params = params
        self.fc_net = UPDeT(input_shape=input_shape, params=params)
        self.action_head = nn.Linear(nr_actions, nr_actions)
        # self.value_head = nn.Linear(nr_actions, 1)
        self.hidden_state = self.fc_net.init_hidden().expand(params["nr_agents"], 1, -1)

    def forward(self, x, h):
        x, _h = self.fc_net(x, h, self.params["nr_agents"], int(self.params["nr_agents"]/2))
        # return F.softmax(self.action_head(x), dim=-1), self.value_head(x)
        return self.action_head(x), _h


class DQNLearner(DeepLearningController):

    def __init__(self, params):
        super(DQNLearner, self).__init__(params)
        self.epsilon = 1.0
        self.epsilon_decay = get_param_or_default(params, "epsilon_decay", 0.0001)
        self.epsilon_min = get_param_or_default(params, "epsilon_min", 0.01)
        self.batch_size = get_param_or_default(params, "batch_size", 64)
        self.ktr = params["ktr"]
        history_length = self.max_history_length
        input_shape = self.input_shape
        num_actions = self.num_actions

        network_constructor = lambda in_shape, actions, length: DQNNet(in_shape, actions, length, params=params)
        #################################################
        # self.encoder_ = PursuitModule(input_shape, num_actions, history_length, network_constructor).to(self.device)
        # self.encoder_optimizer = torch.optim.Adam(self.encoder_.protagonist_parameters(), lr=self.alpha)

        self.policy_net = PursuitModule(input_shape, num_actions, history_length, network_constructor).to(self.device)
        self.target_net = PursuitModule(input_shape, num_actions, history_length, network_constructor).to(self.device)
        if self.ktr:
            self.protagonist_optimizer = KFACOptimizer(self.policy_net.protagonist_net)
            self.protagonist_target_optimizer = KFACOptimizer(self.target_net.protagonist_net)
        else:
            self.protagonist_optimizer = torch.optim.Adam(self.policy_net.protagonist_parameters(), lr=self.alpha)
        self.update_target_network()

    def value_log_action(self, probs, action):
        return Categorical(probs).log_prob(action)
        # log_list = []
        # for i in range(probs.size(0)):
        #     log_list.append(Categorical(probs[i]).log_prob(action[i]))
        # return torch.tensor(log_list).mean()

    def joint_action_probs(self, histories, training_mode=True, agent_ids=None):

        action_probs = []
        used_epsilon = self.epsilon_min
        if training_mode:
            used_epsilon = self.epsilon
        if agent_ids is None:
            agent_ids = self.agent_ids

        for i, agent_id in enumerate(agent_ids):
            # history = [[joint_obs[i]] for joint_obs in histories]
            # history = torch.tensor(history, device=self.device, dtype=torch.float32)
            history = [histories[i]]
            history = torch.tensor(history, device=self.device, dtype=torch.float32)

            Q_values_ = self.policy_net(history)
            Q_values = Q_values_.detach().cpu().numpy()

            assert len(Q_values) == 1, "Expected length 1, but got shape {}".format(Q_values.shape)
            probs = used_epsilon*numpy.ones(self.num_actions)/self.num_actions
            rest_prob = 1 - sum(probs)
            probs[argmax(Q_values[0])] += rest_prob
            action_probs.append(probs/sum(probs))
        return action_probs

    def update(self, state, obs, joint_action, rewards, next_state, next_obs, dones):
        super(DQNLearner, self).update(state, obs, joint_action, rewards, next_state, next_obs, dones)
        if self.warmup_phase <= 0 and self.memory.size() > self.batch_size and dones:
            minibatch = self.memory.sample_batch(self.batch_size)
            minibatch_data = self.collect_minibatch_data(minibatch)
            histories = minibatch_data["pro_histories"]
            next_histories = minibatch_data["next_pro_histories"]
            actions = minibatch_data["pro_actions"]
            rewards = minibatch_data["pro_rewards"]
            self.update_step(histories, next_histories, actions, rewards, self.protagonist_optimizer)
            self.update_target_network()
            self.epsilon = max(self.epsilon - self.epsilon_decay, self.epsilon_min)
            self.training_count += 1
            return True
        return False

    def update_step(self, histories, next_histories, actions, rewards, optimizer):
        action_log_probs = []
        new_actions = torch.clone(actions).view(-1, 1)
        Q_values = self.policy_net(histories)
        for probs, action in zip(Q_values, new_actions):
            action_log = self.value_log_action(probs, action)
            action_log_probs.append(action_log)
        Q_values = Q_values.view(histories.size(0), histories.size(2), -1)
        Q_values = Q_values.gather(2, actions.unsqueeze(2)).squeeze()
        # Q_values = self.policy_net(histories).gather(1, actions.unsqueeze(1)).squeeze()
        next_Q_values= self.target_net(next_histories)
        action_values = torch.clone(next_Q_values)
        next_Q_values = next_Q_values.view(next_histories.size(0), next_histories.size(2), -1)
        next_Q_values = next_Q_values.max(2)[0]
        # next_Q_values = self.target_net(next_histories).max(1)[0].detach()

        action_log_probs = torch.stack(action_log_probs)
        if self.ktr and self.protagonist_optimizer.steps % self.protagonist_optimizer.Ts == 0:
            self.policy_net.protagonist_net.zero_grad()
            pg_fisher_loss = -action_log_probs.mean()
            value_noise = torch.randn(action_values.size())
            if action_values.is_cuda:
                value_noise = value_noise.cuda()
            sample_values = action_values + value_noise
            vf_fisher_loss = -(action_values - sample_values.detach()).pow(2).mean()
            fisher_loss = pg_fisher_loss + vf_fisher_loss
            self.protagonist_optimizer.Acc_stats = True
            fisher_loss.backward(retain_graph=True)
            self.protagonist_optimizer.Acc_stats = False

        target_Q_values = rewards + self.gamma*next_Q_values
        optimizer.zero_grad()
        loss_dqn = F.mse_loss(Q_values, target_Q_values)
        if self.params["use_ml"]:
            a1 = self.policy_net.protagonist_net.encoder_output.tolist()
            a2 = self.policy_net.protagonist_net.fc_output.tolist()
            a = 0
            for _a1, _a2 in zip(a1, a2):
                t_a1 = torch.tensor(_a1)
                t_a2 = torch.tensor(_a2)
                ret = torch.mean(t_a1) - torch.log(torch.mean(torch.exp(t_a2)))
                a -= ret
                # a -= mutual_info_score(_a1, _a2)

            loss_enc = torch.tensor(a, device=self.device, dtype=torch.float32)
            loss = loss_dqn*0.1 + loss_enc*0.9
            self.params["summary_write"].add_scalar("dpn_loss", loss_dqn, self.params["episode_num"])
            self.params["summary_write"].add_scalar("enc_loss", loss_enc, self.params["episode_num"])
        else:
            loss = loss_dqn
            self.params["summary_write"].add_scalar("dpn_loss", loss_dqn, self.params["episode_num"])
        self.params["summary_write"].add_scalar("loss", loss, self.params["episode_num"])
        loss.backward()
        optimizer.step()

        # enc_potimizer.zero_grad()
        # enc_output = self.encoder_(histories)  # 1*48

        return loss
