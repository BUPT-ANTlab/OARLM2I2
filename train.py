import os.path
from os.path import join

import numpy as np
import torch
import numpy
import random
import time
from torch.utils.tensorboard import SummaryWriter
import pandas as pd


def run_episode(episode_id, controller, params, training_mode=True):
    env = params["env"]
    path = params["directory"]
    num_pursuit = params["num_pursuit"]
    controller_eva = params["Eva_Controller"]
    time_step = 0

    pursuit_discounted_return = 0
    pursuit_undiscounted_return = 0

    evader_discounted_return = 0
    evader_undiscounted_return = 0
    last_op_observation_state = None
    last_observation_state = None
    last_joint_action = None
    last_global_state = None
    finally_epoch = 0
    for epoch in range(params["Epoch"]):
        stop_, rewards, eva_rewards = env.step()
        # if params["algorithm_name"] == "DQN" or params["algorithm_name"] == "PPO" or params["algorithm_name"] == "MADDPG":
        if params["train_pur"]:
            pursuit_undiscounted_return += sum(rewards)
            pursuit_discounted_return += (params["gamma"] ** time_step) * sum(rewards)
        else:
            evader_undiscounted_return += sum(eva_rewards)
            evader_discounted_return += (params["gamma"]**time_step)*sum(eva_rewards)
        time_step += 1
        policy_updated = False
        # if epoch > params["Epoch"] - 1 or stop_:
        if epoch > params["Epoch"] - 1:
            stop_ = True
            # env.reset()
            # break
        # else:
        if len(env.pursuit_vehs.keys()) != params["num_pursuit"]:
            # observation_state = env.observation_state
            global_state = env.global_state
            env.pursuitVehControl(choice_random=True)
            env.evadeVehControl(choice="random_choice")

        else:
            op_observation_state = env.op_observation_state
            observation_state = env.observation_state
            global_state = env.global_state
            if training_mode and last_global_state is not None:
                if params["train_pur"]:
                    policy_updated = controller.update(last_global_state, last_observation_state, last_joint_action,
                                                       rewards, global_state, observation_state,
                                                       stop_) or policy_updated
                else:
                    policy_updated = controller.update(last_global_state, last_op_observation_state, last_joint_action,
                                                       eva_rewards, global_state, op_observation_state,
                                                       stop_) or policy_updated
            if stop_:
                env.reset()
                finally_epoch = epoch
                break

            if params["train_pur"]:
                #  这个policy是Deeplearning的policy
                joint_action = controller.policy(observation_state, training_mode)
                # joint_action = np.array([0, 0, 0, 0])
                joint_action = np.array(joint_action)
                env.pursuitVehControl(choice_random=False, commands=joint_action)
                if params["evader_method"] == "Command":
                    eva_joint_action = controller_eva.policy(op_observation_state, training_mode)
                    eva_joint_action = np.array(eva_joint_action)
                    env.evadeVehControl(choice="Command", commands=eva_joint_action)
                else:
                    env.evadeVehControl(choice=params["evader_method"], commands=None)
                last_observation_state = observation_state
            else:
                joint_action = controller.policy(op_observation_state, training_mode)
                # joint_action = np.array([0, 0, 0, 0])
                joint_action = np.array(joint_action)
                env.pursuitVehControl(choice_random=True)
                env.evadeVehControl(choice="Command", commands=joint_action)
                last_op_observation_state = op_observation_state
            last_global_state = global_state
            last_joint_action = joint_action
    if params["train_pur"]:
        return pursuit_discounted_return / time_step, pursuit_undiscounted_return / time_step, policy_updated, finally_epoch
    else:
        return evader_discounted_return/time_step, evader_undiscounted_return/time_step, policy_updated, finally_epoch


def run_test_record(controller, params):
    env = params["env"]
    path = params["directory"]
    num_pursuit = params["num_pursuit"]
    controller_eva = params["Eva_Controller"]
    time_step = 0
    pursuit_discounted_return = 0
    pursuit_undiscounted_return = 0
    pursuit_discounted_records = []
    pursuit_undiscounted_records = []
    evader_discounted_return = 0
    evader_undiscounted_return = 0
    evader_discounted_records = []
    evader_undiscounted_records = []
    for epoch in range(params["Epoch"]):
        stop_, rewards, eva_rewards = env.step()
        # if params["algorithm_name"] == "DQN" or params["algorithm_name"] == "PPO" or params["algorithm_name"] == "MADDPG":
        if params["train_pur"]:
            pursuit_undiscounted_return += sum(eva_rewards)
            pursuit_undiscounted_records.append(sum(eva_rewards))
            pursuit_discounted_return += (params["gamma"] ** time_step) * sum(eva_rewards)
            pursuit_discounted_records.append((params["gamma"] ** time_step) * sum(eva_rewards))
        else:
            evader_undiscounted_return += sum(eva_rewards)
            evader_undiscounted_records.append(sum(eva_rewards))
            evader_discounted_return += (params["gamma"] ** time_step) * sum(eva_rewards)
            evader_discounted_records.append((params["gamma"] ** time_step) * sum(eva_rewards))
        time_step += 1
        if epoch > params["Epoch"] - 1:
            env.reset()
            break
        else:
            if len(env.pursuit_vehs.keys()) != params["num_pursuit"]:
                env.pursuitVehControl(choice_random=True)
                env.evadeVehControl(choice="random_choice")
            else:
                op_observation_state = env.op_observation_state
                observation_state = env.observation_state
                global_state = env.global_state
                if stop_:
                    env.reset()
                    break
                if params["train_pur"]:
                    joint_action = controller.policy(observation_state, training_mode=False)
                    # joint_action = np.array([0, 0, 0, 0])
                    joint_action = np.array(joint_action)
                    env.pursuitVehControl(choice_random=False, commands=joint_action)
                    if params["evader_method"] == "Command":
                        eva_joint_action = controller_eva.policy(op_observation_state, training_mode=False)
                        eva_joint_action = np.array(eva_joint_action)
                        env.evadeVehControl(choice="Command", commands=eva_joint_action)
                    else:
                        env.evadeVehControl(choice=params["evader_method"], commands=None)

                else:
                    joint_action = controller.policy(op_observation_state, training_mode=False)
                    # joint_action = np.array([0, 0, 0, 0])
                    joint_action = np.array(joint_action)
                    env.pursuitVehControl(choice_random=True)
                    env.evadeVehControl(choice="Command", commands=joint_action)

    if params["train_pur"]:
        return pursuit_discounted_return / time_step, pursuit_undiscounted_return / time_step, \
               pursuit_discounted_records, pursuit_undiscounted_records, time_step
    else:
        return evader_discounted_return / time_step, evader_undiscounted_return / time_step,\
               evader_discounted_records, evader_undiscounted_records, time_step


def run_test(num_test_episodes, controller, params):
    num_pursuit = params["num_pursuit"]
    test_discounted_returns = []
    test_undiscounted_returns = []
    finally_epochs = []
    for test_episode_id in range(num_test_episodes):
        print("============> test in episode %s  <============" % (test_episode_id+1))
        discounted_returns, undiscounted_returns, updated, finally_epoch = run_episode(episode_id="Test-{}".format(test_episode_id),
                                                                        controller=controller, params=params,
                                                                        training_mode=False)
        finally_epochs.append(finally_epoch)
        test_discounted_returns.append(discounted_returns)
        test_undiscounted_returns.append(undiscounted_returns)
    return np.mean(test_discounted_returns), np.mean(test_undiscounted_returns), np.mean(finally_epochs)


def run(controller, params):
    path = params["directory"]
    env = params["env"]
    summary_write = SummaryWriter(path)
    params["summary_write"] = summary_write
    num_test_episodes = params["test_episodes"]
    training_discounted_returns = []
    training_undiscounted_returns = []
    test_discounted_returns = []
    test_undiscounted_returns = []
    test_discounted_return, test_undiscounted_return, finally_epochs = run_test(num_test_episodes, controller, params)
    test_discounted_returns.append(test_discounted_return)
    test_undiscounted_returns.append(test_undiscounted_return)
    epoch_updates = 0
    best_finally_epoch = 7200
    best_discounted_return = -1000
    test_best_discounted_return = -1000
    test_best_undiscounted_return = -1000
    if params["train_pur"]:
        best_epoch = 7200
    else:
        best_epoch = 0
    for episode in range(params["Episode"]):
        params["episode_num"] = episode
        print("============> run in episode %s <============" % episode)
        discounted_returns, undiscounted_returns, updated, finally_epoch = run_episode(episode, controller, params)
        training_discounted_returns.append(discounted_returns)
        training_undiscounted_returns.append(undiscounted_returns)
        if updated:
            print("============> updated in episode", epoch_updates, "<============")
            if epoch_updates % 5 == 0:
                test_discounted_return, test_undiscounted_return, finally_epochs = run_test(num_test_episodes, controller, params)
                test_discounted_returns.append(test_discounted_return)
                test_undiscounted_returns.append(test_undiscounted_return)
                summary_write.add_scalar("test_discounted_return", test_discounted_return, int(epoch_updates/1))
                summary_write.add_scalar("test_undiscounted_return", test_undiscounted_return, int(epoch_updates/1))
                summary_write.add_scalar("test_total_epoch", finally_epochs, int(epoch_updates/1))
                # if test_discounted_return > best_discounted_return:
                #     print("Save the best example, epoch is %s" % epoch_updates)
                #     # best_finally_epoch = finally_epoch
                #     best_discounted_return = test_discounted_return
                #     best_path = join(path, "best.pth")
                #     controller.save_weights_to_path(best_path)
            # if epoch_updates % 20 == 0:
                # if best_finally_epoch > finally_epoch:
            if test_discounted_return > test_best_discounted_return and finally_epochs < best_epoch:
                if params["algorithm_name"] == "Q-Learning":
                    print("Save the best example, epoch is %s" % epoch_updates)
                    controller.q_table.to_csv(join(path, "best.csv"))
                else:
                    print("Save the best example, epoch is %s" % epoch_updates)
                    # best_finally_epoch = finally_epoch
                    test_best_discounted_return = test_discounted_return
                    best_epoch = finally_epochs
                    summary_write.add_scalar("test best epoch", best_epoch, int(epoch_updates/1))
                    summary_write.add_scalar("test best discounted return", test_best_discounted_return, int(epoch_updates/1))
                    best_path = join(path, "best.pth")
                    controller.save_weights_to_path(best_path)
            if test_undiscounted_return > test_best_undiscounted_return:
                test_best_undiscounted_return = test_undiscounted_return
                summary_write.add_scalar("test best undiscounted return", test_best_undiscounted_return, int(epoch_updates/1))
            if epoch_updates % 20 == 0:
                if params["algorithm_name"] == "Q-Learning":
                    pass
                else:
                    if os.path.exists(join(path, "best.pth")):
                        controller.load_weights_from_history(join(path, "best.pth"))
            epoch_updates += 1
        summary_write.add_scalar("discounted_return", discounted_returns, episode)
        summary_write.add_scalar("undiscounted_return", undiscounted_returns, episode)
    return True


def test(controller, params, testEpoch):
    temp = 0
    for test_epoch in range(testEpoch):
        discounted_return, undiscounted_return, discounted_records, undiscounted_records, step = run_test_record(controller, params)
        print("Undiscounted return", undiscounted_return, "Step", step)
        # if undiscounted_return > temp:
        #     temp = undiscounted_return
        #     record = np.array(undiscounted_records)
        #     data1 = pd.DataFrame(record)
        #     data1.to_csv(params["test_output_csv"])

        # if step > 194 and undiscounted_return > 0:
        #     temp = undiscounted_return
        #     record = np.array(undiscounted_records)
        #     data1 = pd.DataFrame(record)
        #     data1.to_csv(params["test_output_csv"])

