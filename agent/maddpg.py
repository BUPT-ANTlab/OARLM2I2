from agent.ppo import PPONet
from agent.controller import DeepLearningController
from agent.utils import get_param_or_default
from agent.modules import MLP, PursuitModule
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy


class CriticNet(nn.Module):
    def __init__(self, nr_actions, nr_agents, state_shape, nr_hidden_layers=128, params=None):
        super(CriticNet, self).__init__()
        self.nr_actions = nr_actions
        self.nr_agents = nr_agents
        self.global_input_shape = numpy.prod(state_shape)
        self.joint_action_dim = int(self.nr_actions*self.nr_agents)

        # Set up network layers
        self.batchnorm_state = nn.BatchNorm1d(int(self.global_input_shape/4))
        self.fc_state = nn.Linear(int(self.global_input_shape/4), int(nr_hidden_layers/2))
        self.fc_actions = nn.Linear(self.joint_action_dim, int(nr_hidden_layers/2))
        self.fc2 = nn.Linear(nr_hidden_layers, nr_hidden_layers)
        self.fc3 = nn.Linear(nr_hidden_layers, 1)

    def forward(self, states, actions, device):
        states, joint_actions = self.build_inputs(states, actions, device)
        x1 = F.elu(self.fc_state(self.batchnorm_state(states)))
        x2 = F.elu(self.fc_actions(joint_actions))
        x = torch.cat([x1, x2], dim=-1)
        x = F.elu(self.fc2(x))
        return self.fc3(x)

    def build_inputs(self, states, actions, device):
        batch_size = states.size(0)
        states = states.view(batch_size, -1)
        actions = actions.view(batch_size, -1)
        return states, actions


class MADDPGLearner(DeepLearningController):

    def __init__(self, params):
        super(MADDPGLearner, self).__init__(params)
        self.batch_size = params["batch_size"]
        self.nr_epochs = 5
        self.minibatch_size = 32
        self.nr_episodes = get_param_or_default(params, "nr_episodes", 10)
        self.warmup_phase_epochs = 16
        self.minimax = params["minimax"]
        self.pertubation_rate = get_param_or_default(params, "pertubation_rate", 0.01)
        self.epsilon = 1.0
        self.epsilon_decay = 1.0/50
        self.epsilon_min = 0.01
        history_length = self.max_history_length
        input_shape = self.input_shape
        num_actions = self.num_actions
        self.tau = 0.01
        network_constructor = lambda in_shape, actions, length: PPONet(in_shape, actions, length, False, params=self.params)
        self.policy_net = PursuitModule(input_shape, num_actions, history_length, network_constructor).to(self.device)
        self.protagonist_optimizer = torch.optim.Adam(self.policy_net.protagonist_parameters(), lr=self.alpha)
        self.target_policy_net = PursuitModule(input_shape, num_actions, history_length, network_constructor).to(self.device)

        self.protagonist_critic_net = CriticNet(self.num_actions, self.num_agents, self.global_input_shape, params=self.params).to(self.device)
        self.target_protagonist_critic_net = CriticNet(self.num_actions, self.num_agents, self.global_input_shape, params=self.params).to(self.device)
        self.protagonist_critic_optimizer = torch.optim.Adam(self.protagonist_critic_net.parameters(), lr=self.alpha)
        self.protagonist_target_critic_optimizer = torch.optim.Adam(\
            list(self.target_policy_net.protagonist_net.parameters()) +
            list(self.target_protagonist_critic_net.parameters()), lr=self.alpha)

        self.target_nets = [self.target_policy_net.protagonist_net, self.target_protagonist_critic_net]
        self.original_nets = [self.policy_net.protagonist_net, self.protagonist_critic_net]
        self.reset_target_networks()

    def reset_target_networks(self):
        for target_net, original_net in zip(self.target_nets, self.original_nets):
            target_net.load_state_dict(original_net.state_dict())
            target_net.eval()

    def update_target_networks(self):
        for target_net, original_net in zip(self.target_nets, self.original_nets):
            for target_param, param in zip(target_net.parameters(), original_net.parameters()):
                target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

    def value_update(self, minibatch_data):
        batch_size = minibatch_data["states"].size(0)
        states = minibatch_data["states"]
        next_states = minibatch_data["next_states"]
        next_states = torch.tensor(next_states, dtype=torch.float32, device=self.device)
        zeros = torch.zeros(batch_size, dtype=torch.long, device=self.device).unsqueeze(1)
        next_histories = minibatch_data["next_pro_histories"].view(batch_size*self.num_agents, 4,4,33)
        rewards = minibatch_data["pro_rewards"].view(batch_size, -1).gather(1, zeros).squeeze()
        actions = self.actions_to_one_hot(minibatch_data["pro_actions"])
        critic = self.protagonist_critic_net
        target_critic = self.target_protagonist_critic_net
        optimizer = self.protagonist_critic_optimizer
        target_optimizer = self.protagonist_target_critic_optimizer
        if self.minimax:
            agent_index = numpy.random.randint(0, self.num_agents)
            next_actions, _ = self.target_policy_net(next_histories, use_gumbel_softmax=True)
            next_actions = torch.tensor(next_actions.detach().cpu().numpy(), device=self.device, dtype=torch.float32, requires_grad=True)
            target_optimizer.zero_grad()
            target_loss = -1.0*target_critic(next_states, next_actions, self.device).mean()
            target_loss.backward()
            gradients = next_actions.grad.detach()
            gradients = self.pertubation_rate*gradients
            for gradient in gradients.view(batch_size, -1):
                index = int(agent_index*self.num_actions)
                for i in range(self.num_actions):
                    gradient[index+i] = 0
            next_actions = next_actions.detach() + gradients
            Q_targets = target_critic(next_states, next_actions, self.device).squeeze()
        Q_targets = Q_targets.detach()
        Q_targets = rewards + self.gamma*Q_targets
        Q_targets = Q_targets.detach().squeeze()
        Q_values = critic(states, actions, self.device)
        Q_values = Q_values.squeeze()
        optimizer.zero_grad()
        loss = F.mse_loss(Q_values, Q_targets)
        loss.backward()
        optimizer.step()

    def actions_to_one_hot(self, actions):
        actions = actions.view(actions.size(0)*actions.size(1), -1)
        actions = actions.detach().cpu().numpy()
        one_hots = numpy.zeros((len(actions), self.num_actions))
        for index, action in enumerate(actions):
            one_hots[index][action[0]] = 1
        return torch.tensor(one_hots, dtype=torch.float32, device=self.device)

    def joint_action_probs(self, histories, training_mode=True, agent_ids=None):
        action_probs = []
        if agent_ids is None:
            agent_ids = self.agent_ids
        if self.warmup_phase_epochs > 0:
            return [numpy.ones(self.num_actions)/self.num_actions for _ in agent_ids]
        else:
            for i, agent_id in enumerate(agent_ids):
                history = [histories[i]]
                history = torch.tensor(history, device=self.device, dtype=torch.float32)
                if numpy.random.rand() <= self.epsilon:
                    probs = numpy.zeros(self.num_actions)
                    probs[numpy.random.randint(0, self.num_actions)] = 1
                else:
                    probs, value = self.policy_net(history, use_gumbel_softmax=training_mode)
                    assert len(probs) == 1, "Expected length 1, but got shape {}".format(probs.shape)
                    probs = probs.detach().cpu().numpy()[0]
                    value = value.detach()
                action_probs.append(probs)
        return action_probs

    def policy_update(self, minibatch_data, optimizer, random_agent_index=None):
        policy_loss = 0
        warmup_phase_over = self.warmup_phase_epochs <= 0
        if warmup_phase_over:
            states = minibatch_data["states"]
            batch_size = states.size(0)
            histories = minibatch_data["pro_histories"]
            critic = self.protagonist_critic_net
            num_agents = self.num_agents
            action_probs, _ = self.policy_net(histories, use_gumbel_softmax=True)
            one_hots = action_probs.clone().view(batch_size, num_agents, self.num_actions).detach()
            action_probs = action_probs.view(batch_size, num_agents, self.num_actions)
            # index = numpy.random.randint(0, num_agents)
            # for joint_action1, joint_action2 in zip(one_hots, action_probs):
            #     joint_action1[index] = joint_action2[index]
            Q_values = critic(states, action_probs, self.device)
            loss = -1.0*Q_values.mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        return policy_loss

    def update(self, state, observations, joint_action, rewards, next_state, next_observations, dones):
        super(MADDPGLearner, self).update(state, observations, joint_action, rewards, next_state, next_observations, dones)
        global_terminal_reached = dones
        # if global_terminal_reached and self.memory.size() >= self.nr_episodes:
        if self.warmup_phase <= 0 and self.memory.size() > self.batch_size and dones:
            is_protagonist = True
            trainable_setting = is_protagonist
            if trainable_setting:
                for _ in range(self.nr_epochs):
                    batch = self.memory.sample_batch(self.minibatch_size)
                    minibatch_data = self.collect_minibatch_data(batch, whole_batch=True)
                    self.value_update(minibatch_data)
                    optimizer = self.protagonist_optimizer
                    self.policy_update(minibatch_data, optimizer)
                    self.update_target_networks()
                if self.warmup_phase_epochs <= 0:
                    self.epsilon = max(self.epsilon_min, self.epsilon - self.epsilon_decay)
                self.warmup_phase_epochs -= 1
                self.warmup_phase_epochs = max(0, self.warmup_phase_epochs)
            # self.memory.clear()
            return True
        return False
