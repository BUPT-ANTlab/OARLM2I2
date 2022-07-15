import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from agent.utils import argmax, get_param_or_default
from agent.controller import DeepLearningController, ReplayMemory
from agent.modules import MLP, PursuitModule, UPDeT, MLPencoder, EncOut
from agent.kfac import KFACOptimizer


class PPONet(nn.Module):
    def __init__(self, input_shape, num_actions, max_history_length, q_values=False, params=None):
        super(PPONet, self).__init__()
        self.params = params
        if params["use_encoder"]:
            self.enc_out_net = EncOut((1, 132))
            self.encoder_net = MLPencoder(input_shape, max_history_length, params=params)
        self.fc_net = MLP(input_shape, max_history_length, params=params)
        self.action_head = nn.Linear(self.fc_net.nr_hidden_units, num_actions)
        if q_values:
            self.value_head = nn.Linear(self.fc_net.nr_hidden_units, num_actions)
        else:
            self.value_head = nn.Linear(self.fc_net.nr_hidden_units, 1)

    def forward(self, x, use_gumbel_softmax=False):
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
                if use_gumbel_softmax:
                    return F.gumbel_softmax(self.action_head(x), hard=True, dim=-1), self.value_head(x)
                return F.softmax(self.action_head(x), dim=-1), self.value_head(x)
            else:
                current_x = x[:, -1]
                current_x = current_x.view(x.size(0), -1)
                x = current_x
                x = self.fc_net(x)
                if use_gumbel_softmax:
                    return F.gumbel_softmax(self.action_head(x), hard=True, dim=-1), self.value_head(x)
                return F.softmax(self.action_head(x), dim=-1), self.value_head(x)
        else:
            if self.params["use_encoder"]:
                self.encoder_output = self.encoder_net(x[:, 0:3])  # 取前三个输入；输出是1*48
                current_x = x[:, -1]
                current_x = current_x.view(x.size(0), -1)  # 1*(4*33)
                self.fc_output = self.enc_out_net(current_x)  # 输入一个全连接层，得到1*48
                x = torch.cat([current_x, self.encoder_output], 1)
                x = self.fc_net(x)
                if use_gumbel_softmax:
                    return F.gumbel_softmax(self.action_head(x), hard=True, dim=-1), self.value_head(x)
                return F.softmax(self.action_head(x), dim=-1), self.value_head(x)
            else:
                current_x = x[:, -1]
                current_x = current_x.view(x.size(0), -1)  # 1*(4*33)
                x = current_x
                x = self.fc_net(x)
                if use_gumbel_softmax:
                    return F.gumbel_softmax(self.action_head(x), hard=True, dim=-1), self.value_head(x)
                return F.softmax(self.action_head(x), dim=-1), self.value_head(x)


class PPOLearner(DeepLearningController):

    def __init__(self, params):
        super(PPOLearner, self).__init__(params)
        self.nr_epochs = get_param_or_default(params, "nr_epochs", 5)
        self.nr_episodes = get_param_or_default(params, "nr_episodes", 10)
        self.eps_clipping = get_param_or_default(params, "eps_clipping", 0.2)
        self.use_q_values = get_param_or_default(params, "use_q_values", False)
        self.batch_size = get_param_or_default(params, "batch_size", 64)
        self.epsilon_min = get_param_or_default(params, "epsilon_min", 0.01)
        history_length = self.max_history_length
        input_shape = self.input_shape
        num_actions = self.num_actions
        network_constructor = lambda in_shape, actions, length: PPONet(in_shape, actions, length, self.use_q_values, params)
        self.policy_net = PursuitModule(input_shape, num_actions, history_length, network_constructor).to(self.device)
        self.ktr = params["ktr"]
        if self.ktr:
            self.protagonist_optimizer = KFACOptimizer(self.policy_net.protagonist_net)
        else:
            self.protagonist_optimizer = torch.optim.Adam(self.policy_net.protagonist_parameters(), lr=self.alpha)

    def joint_action_probs(self, histories, training_mode=True, agent_ids=None):
        action_probs = []
        used_epsilon = self.epsilon_min
        if training_mode:
            used_epsilon = self.epsilon
        if agent_ids is None:
            agent_ids = self.agent_ids

        for i, agent_id in enumerate(agent_ids):
            history = [histories[i]]
            history = torch.tensor(history, device=self.device, dtype=torch.float32)
            probs, value = self.policy_net(history)
            # print("1Once time is %s" % (time.time() - in_time))
            assert len(probs) == 1, "Expected length 1, but got shape {}".format(probs.shape)
            probs = probs.detach().cpu().numpy()[0]
            value = value.detach()
            action_probs.append(probs)
        return action_probs

    def value_log_action(self, probs, action):
        return Categorical(probs).log_prob(action)

    def policy_update(self, minibatch_data, optimizer, random_agent_indices=None):
        returns = minibatch_data["pro_returns"]
        actions = minibatch_data["pro_actions"]
        old_probs = minibatch_data["pro_action_probs"]
        histories = minibatch_data["pro_histories"]

        # reshape the size of the actions and returns 2022/06/15
        actions = actions.view(-1, 1)
        returns = returns.view(-1, 1)

        action_probs, expected_values = self.policy_net(histories)
        policy_losses = []
        value_losses = []
        action_log_probs = []
        values = []
        for probs, action, value, R, old_prob in zip(action_probs, actions, expected_values, returns, old_probs):
            value_index = 0
            if self.use_q_values:
                value_index = action
                advantage = value[value_index].detach()
            else:
                advantage = R - value.item()
            values.append(value)
            action_log = self.value_log_action(probs, action)
            action_log_probs.append(action_log)
            policy_losses.append(self.policy_loss(advantage, probs, action, old_prob))
            value_losses.append(F.mse_loss(value[value_index], torch.tensor(R)))
        value_loss = torch.stack(value_losses).mean()
        policy_loss = torch.stack(policy_losses).mean()
        action_log_probs = torch.stack(action_log_probs)
        values = torch.stack(values)
        self.params["summary_write"].add_scalar("value_loss", value_loss, self.params["episode_num"])
        self.params["summary_write"].add_scalar("policy_loss", policy_loss, self.params["episode_num"])
        if self.ktr and self.protagonist_optimizer.steps % self.protagonist_optimizer.Ts == 0:
            self.policy_net.protagonist_net.zero_grad()
            pg_fisher_loss = -action_log_probs.mean()
            value_noise = torch.randn(values.size())
            if values.is_cuda:
                value_noise = value_noise.cuda()
            sample_values = values + value_noise
            vf_fisher_loss = -(values - sample_values.detach()).pow(2).mean()
            fisher_loss = pg_fisher_loss + vf_fisher_loss
            self.protagonist_optimizer.Acc_stats = True
            fisher_loss.backward(retain_graph=True)
            self.protagonist_optimizer.Acc_stats = False
        loss = policy_loss + value_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return True

    def policy_loss(self, advantage, probs, action, old_prob):
        m1 = Categorical(probs)
        m2 = Categorical(old_prob)
        ratio = torch.exp(m1.log_prob(action) - m2.log_prob(action))
        clipped_ratio = torch.clamp(ratio, 1-self.eps_clipping, 1+self.eps_clipping)
        surrogate_loss1 = ratio*advantage
        surrogate_loss2 = clipped_ratio*advantage
        return -torch.min(surrogate_loss1, surrogate_loss2)

    def value_update(self, minibatch_data):
        pass

    def update(self, state, observations, joint_action, rewards, next_state, next_observations, dones):
        super(PPOLearner, self).update(state, observations, joint_action, rewards, next_state, next_observations, dones)
        global_terminal_reached = dones
        # if global_terminal_reached and self.memory.size() > self.nr_episodes:
        if self.warmup_phase <= 0 and self.memory.size() > self.batch_size and dones:
            # 默认 capacity为 20000
            batch_size = self.batch_size if self.batch_size < self.memory.capacity else self.memory.capacity
            batch = self.memory.sample_batch(batch_size)

            minibatch_data = self.collect_minibatch_data(batch, whole_batch=True)

            self.value_update(minibatch_data)
            for _ in range(self.nr_epochs):
                optimizer = self.protagonist_optimizer
                self.policy_update(minibatch_data, optimizer)
            # self.memory.clear()
            return True
        return False
