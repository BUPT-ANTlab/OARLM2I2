import numpy as np

from agent.ppo import PPOLearner
from torch.distributions import Categorical
from agent.kfac import KFACOptimizer
import torch
import torch.nn as nn
import torch.nn.functional as F


class A2CLearner(PPOLearner):

    def __init__(self, params):
        params["nr_epochs"] = 1
        super(A2CLearner, self).__init__(params)
        self.ktr = params["ktr"]
        if params["ktr"]:
            self.protagonist_optimizer = KFACOptimizer(self.policy_net.protagonist_net)

    def policy_loss(self, advantage, probs, action, old_prob):
        # return -1 * self.value_log_action(probs, action) * advantage
        m = Categorical(probs)
        return -m.log_prob(action) * advantage

    def value_log_action(self, probs, action):
        return Categorical(probs).log_prob(action)
        # log_list = []
        # for i in range(probs.size(0)):
        #     log_list.append(Categorical(probs[i]).log_prob(action[i]))
        # return torch.tensor(log_list).mean()

    def policy_update(self, minibatch_data, optimizer, random_agent_indices=None):
        returns = minibatch_data["pro_returns"]
        actions = minibatch_data["pro_actions"]
        old_probs = minibatch_data["pro_action_probs"]
        histories = minibatch_data["pro_histories"]

        actions = actions.view(-1, 1)
        returns = returns.view(-1, 1)

        action_probs, expected_values = self.policy_net(histories)
        # action_probs = action_probs.view(-1, self.params["num_pursuit"], action_probs.size(-1))
        # expected_values = expected_values.view(-1, self.params["num_pursuit"])
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
                # advantage = R.mean() - value.mean().item()
            values.append(value)
            action_log = self.value_log_action(probs, action)
            action_log_probs.append(action_log)
            # action_log_probs.append(Categorical(probs).log_prob(action))
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