import random
import numpy
import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from agent.utils import get_param_or_default
from agent.ppo import PPOLearner

class PPOMIXLearner(PPOLearner):

    def __init__(self, params):
        self.global_input_shape = params["global_observation_shape"]
        super(PPOMIXLearner, self).__init__(params)
        self.central_q_learner = params["central_q_learner"]
        self.last_q_loss = 0

    def value_update(self, minibatch_data):
        batch_size = minibatch_data["states"].size(0)
        self.central_q_learner.zero_actions = torch.zeros(batch_size, dtype=torch.long, device=self.device).unsqueeze(1)
        nr_agents = self.params["num_pursuit"]

        returns = minibatch_data["pro_returns"].view(-1, nr_agents)
        returns = returns.gather(1, self.central_q_learner.zero_actions).squeeze()
        returns /= nr_agents
        returns *= nr_agents
        assert returns.size(0) == batch_size
        for _ in range(self.nr_epochs):
            self.last_q_loss = self.central_q_learner.train_step_with(minibatch_data, returns, nr_agents)

    def policy_update(self, minibatch_data, optimizer, random_agent_indices=None):
        old_probs = minibatch_data["pro_action_probs"]
        histories = minibatch_data["pro_histories"]
        actions = minibatch_data["pro_actions"]
        returns = minibatch_data["pro_returns"]

        actions = actions.view(-1, 1)
        returns = returns.view(-1, 1)

        action_probs, expected_values = self.policy_net(histories)
        expected_Q_values = self.central_q_learner.policy_net(histories).detach()
        policy_losses = []
        value_losses = []
        action_log_probs = []
        values = []
        for probs, action, value, Q_values, old_prob, R in\
            zip(action_probs, actions, expected_values, expected_Q_values, old_probs, returns):
            baseline = sum(probs*Q_values)
            baseline = baseline.detach()
            advantage = R - baseline
            values.append(value)
            action_log = self.value_log_action(probs, action)
            action_log_probs.append(action_log)
            policy_losses.append(self.policy_loss(advantage, probs, action, old_prob))
            value_losses.append(F.mse_loss(value[0], Q_values[action]))
        policy_loss = torch.stack(policy_losses).mean()
        value_loss = torch.stack(value_losses).mean()
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
