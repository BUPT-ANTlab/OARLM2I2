import torch
import agent.dqn as dqn
import agent.ppo as ppo
import agent.maddpg as maddpg
import agent.qmix as qmix
import agent.a2c as a2c
import agent.ppomix as ppomix
import agent.a2cmix as a2cmix
import agent.qlearning as ql

def make(algorithm, params):
    if algorithm == "DQN":
        return dqn.DQNLearner(params)

    if algorithm == "PPO":
        return ppo.PPOLearner(params)

    if algorithm == "MADDPG":
        return maddpg.MADDPGLearner(params)

    if algorithm == "QMIX":
        return qmix.QMIXLearner(params)

    if algorithm == "A2C":
        return a2c.A2CLearner(params)

    if algorithm == "PPO-QMIX":
        params["central_q_learner"] = qmix.QMIXLearner(params)
        return ppomix.PPOMIXLearner(params)

    if algorithm == "A2C-QMIX":
        params["central_q_learner"] = qmix.QMIXLearner(params)
        return a2cmix.A2CMIXLearner(params)

    if algorithm == "Q-Learning":
        return ql.QLearning(params)

