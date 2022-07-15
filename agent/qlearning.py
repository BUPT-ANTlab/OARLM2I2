import numpy as np
import pandas

from agent.controller import Controller
import pandas as pd


def calculateBinToNum(binList):
    result = 0
    for index, item in enumerate(binList[::-1]):
        result += pow(2, index)*item
    return result


def getBinNums(MaxLen):
    BinNums = []
    for i_1 in range(3, MaxLen):
        for i_2 in range(2, i_1):
            for i_3 in range(1, i_2):
                for i_4 in range(0, i_3):
                    BinNums.append(pow(2, i_1) + pow(2, i_2) + pow(2, i_3) + pow(2, i_4))
    for i_1 in range(2, MaxLen):
        for i_2 in range(1, i_1):
            for i_3 in range(0, i_2):
                BinNums.append(pow(2, i_1) + pow(2, i_2) + pow(2, i_3))
    for i_1 in range(1, MaxLen):
        for i_2 in range(0, i_1):
            BinNums.append(pow(2, i_1) + pow(2, i_2))
    for i_1 in range(0, MaxLen):
        BinNums.append(pow(2, i_1))
    BinNums.append(0)
    return BinNums


class QLearning(Controller):
    def __init__(self, params):
        super(QLearning, self).__init__(params)
        self.BinNums = []
        if params["global_ob"]:
            self.q_table = self.build_q_table(48, [0, 1, 2])
        else:
            self.q_table = self.build_q_table(16, [0, 1, 2])
        self.epsilon = 0.9

    def build_q_table(self, n_states, actions):
        # 将state的
        self.BinNums = getBinNums(n_states)
        table = pd.DataFrame(
            np.zeros((len(self.BinNums), len(actions))),
            columns=actions)
        table.index = self.BinNums
        # a = table.index
        # index_num = self.BinNums.index(274877906944)
        # b = table.iloc[index_num,:]
        return table

    def load_weights_from_history(self, path):
        table = pandas.read_csv(path, index_col=0)
        self.q_table = table

    def policy(self, observations, training_mode=True):
        new_observations = self.change_observation(observations)
        joint_action = []
        for new_observation in new_observations:
            # if new_observation not in self.q_table.index:
            #     a = 1
            new_observation = self.BinNums.index(new_observation)
            state_actions = self.q_table.iloc[new_observation, :]
            if (np.random.uniform() > self.epsilon) or (state_actions.all() == 0):
                action = np.random.choice(self.actions)
            else:
                action = state_actions.idxmax()
            joint_action.append(action)
        return joint_action

    def update(self, state, observations, joint_action, rewards, next_state, next_observations, dones):
        if len(rewards) == 0:
            return True
        if len(observations) != len(rewards):
            if self.params["global_ob"]:
                for _ in range(abs(len(observations)-len(rewards))):
                    observations.pop()
            else:
                index_list = []
                for index, observation in enumerate(observations):
                    if observation[0] in [x[0] for x in next_observations]:
                        continue
                    else:
                        index_list.append(index)
                for index, pos in enumerate(index_list):
                    observations.pop(pos-index)
        new_observations = self.change_observation(observations)
        new_next_observations = self.change_observation(next_observations)

        for index, new_observation in enumerate(new_observations):
            q_predict = self.q_table.loc[new_observation, joint_action[index]]

            if dones:
                q_target = rewards[index]
            else:
                reward_index = rewards[index]
                state_index = new_next_observations[index]
                state_index = self.BinNums.index(state_index)
                q_target = reward_index + self.gamma * self.q_table.iloc[state_index, :].max()

            self.q_table.loc[new_observation, joint_action[index]] += self.alpha*(q_target - q_predict)
        return True

    def change_observation(self, observations):
        new_observations = []
        if self.params["global_ob"]:
            for observation in observations:
                new_observation_num = calculateBinToNum(observation)
                new_observations.append(new_observation_num)
        else:
            for eva_id in self.params["evader_ids"]:
                if eva_id in [x[0] for x in observations]:
                    new_observation = []
                    for observation in observations:
                        if observation[0] == eva_id:
                            new_observation += observation[1:]
                    for observation in observations:
                        if observation[0] != eva_id:
                            new_observation += observation[1:]
                    new_observation_num = calculateBinToNum(new_observation)
                    new_observations.append(new_observation_num)

        return new_observations

