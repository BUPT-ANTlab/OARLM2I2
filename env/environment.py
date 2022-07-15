import copy
import time

import numpy as np
import traci
from env.utils import generate_topology, get_junction_links, get_adj, get_bin
import env.utils as utils
import random
import traci
import subprocess
import sys
import logging
import heapq
import platform

# sys.path.append(r'./Informer-MVP')
# from predict import MVP_Informer
# sys.path.append(r'./CNN')
# from cnn_predict import cnn_predict

if platform.system().lower() == 'windows':
    sumoBinary = "D:\\Download\\Sumo\\bin\\sumo-gui"
    sumoBinary_nogui = "D:\\Download\\Sumo\\bin\\sumo"
elif platform.system().lower() == 'linux':
    sumoBinary = "/usr/share/sumo/bin/sumo-gui"
    sumoBinary_nogui = "/usr/share/sumo/bin/sumo"


# 车辆随机选择下一节点
def random_select_next_lane(_next_lanes):
    num_list = list(range(len(_next_lanes)))
    next_lane = random.choice(num_list)
    return list(_next_lanes)[next_lane]


def generate_dict_lane_num(lane_keys):
    lane_to_num = {}
    num = 0
    for key in lane_keys:
        lane_to_num[key] = num
        num += 1
    return lane_to_num


def get_turn_lane(lane_links):
    turn_term = {"l": None,
                 "s": None,
                 "r": None}
    for i in range(len(lane_links)):
        lane_link = lane_links[i]
        edge = lane_link[0].split("_")[0]
        turn_term[lane_link[6]] = edge
    return turn_term


def get_action(current_lane, action):
    action_trans = {0: "l",
                    1: "s",
                    2: "r"}
    current_lane_links = traci.lane.getLinks(current_lane)
    turn_term = get_turn_lane(current_lane_links)
    turn_str = action_trans[int(action)]
    turn_action = turn_term[turn_str]
    next_edge = None
    action_true = False
    if turn_action is not None:
        next_edge = turn_action
        action_true = True
    else:
        for turn_other in ["l", "s", "r"]:
            if turn_term[turn_other] is not None:
                next_edge = turn_term[turn_other]
                break
            else:
                continue
    return next_edge, action_true


class Environment:
    # 初始化环境
    def __init__(self, params):
        self.PORT = params["port"]
        self.rou_path = params["rou_path"]
        self.cfg_path = params["cfg_path"]
        self.net_path = params["net_path"]
        self.params = params
        self.topology, self.node_pos = generate_topology(net_xml_path=self.net_path)
        adj = self.topology.adj
        self.lane2num = generate_dict_lane_num(adj)
        # 记录所有车辆信息
        self.vehicle_list = {}
        # 记录每条路上车辆的数目
        self.lane_vehs = {}
        # 记录追逐车辆的信息
        self.pursuit_vehs = {}
        # 记录逃避车辆的信息
        self.evader_vehs = {}
        # 记录全局信息
        self.global_state = []
        # 记录追逐每个车辆的观测信息
        self.observation_state_ = []

        self.op_observation_state = []

        # 联合观测信息（不加encoder）
        self.observation_state = self.initState_his()
        self.nr_history = 4


        # 启动仿真环境
        self.sumoProcess = self.simStart()
        self.laneIDList = traci.lane.getIDList()
        self.junctionLinks, self.laneList = get_junction_links(self.laneIDList)
        self.adj = np.array(get_adj(self.topology))
        self.adj[self.adj > 0] = 1
        self.params["adj_matrix"] = self.adj
        self.vehicles = traci.vehicle.getIDList()

        # # 预测初始化
        # if self.params["pre_method"] == "informer":
        #     self.pr = MVP_Informer()
        # else:
        #     self.pr = cnn_predict(load_path="./CNN/best_weights_cnn.hdf5")
        # self.initPre()

    def simStart(self):
        if self.params["gui"]:
            sumoProcess = subprocess.Popen(
                [sumoBinary, "-c", self.cfg_path, "--remote-port", str(self.PORT), "--start"],
                stdout=sys.stdout, stderr=sys.stderr)
        else:
            sumoProcess = subprocess.Popen(
                [sumoBinary_nogui, "-c", self.cfg_path, "--remote-port", str(self.PORT), "--start"],
                stdout=sys.stdout, stderr=sys.stderr)
        traci.init(self.PORT)

        logging.info("start TraCI.")
        return sumoProcess

    def reset(self):
        traci.close()
        self.sumoProcess.kill()
        # ==============================重置状态信息=================================
        # 记录所有车辆信息
        self.vehicle_list = {}
        # 记录每条路上车辆的数目
        self.lane_vehs = {}
        # 记录追逐车辆的信息
        self.pursuit_vehs = {}
        # 记录逃避车辆的信息
        self.evader_vehs = {}
        self.sumoProcess = self.simStart()
        self.vehicles = traci.vehicle.getIDList()
        self.state_histories = self.initState_his()
        # if self.params["pre_method"] == "informer":
        #     # self.pr = MVP_Informer()
        #     self.initPre()
        # else:
        #     self.initPre()
        #     # self.pr = cnn_predict(load_path="./CNN/best_weights_cnn.hdf5")
        # self.initPre()

    def step(self):
        traci.simulationStep()
        self.vehicles = traci.vehicle.getIDList()

        for vehicle in self.vehicles:
            self.vehicle_list[vehicle] = {"routeLast": traci.vehicle.getRoute(vehicle)[-1]}
            if "p" in vehicle:
                p_x, p_y = traci.vehicle.getPosition(vehicle)
                p_lane = traci.vehicle.getLaneID(vehicle)
                p_lane = self.checkLane(p_lane)
                next_lane_links = traci.lane.getLinks(p_lane)
                p_turn_term = get_turn_lane(next_lane_links)
                p_lane_position = traci.vehicle.getLanePosition(vehicle)
                p_target = traci.vehicle.getRoute(vehicle)[-1]
                if vehicle not in self.pursuit_vehs.keys():
                    self.pursuit_vehs[vehicle] = {"x": p_x,
                                                  "y": p_y,
                                                  "p_lane": p_lane,
                                                  "p_edge": p_lane.split("_")[0],
                                                  "p_lane_left": p_turn_term["l"],
                                                  "p_lane_straight": p_turn_term["s"],
                                                  "p_lane_right": p_turn_term["r"],
                                                  "p_lane_position": p_lane_position,
                                                  "p_target": p_target,
                                                  "target_evader": None,
                                                  "target_evader_dis": 100,
                                                  "target_evader_dis_last": 100,
                                                  "num_capture": 0}
                else:
                    target_evader = None
                    target_evader_dis = self.pursuit_vehs[vehicle]["target_evader_dis"]
                    target_evader_dis_last = self.pursuit_vehs[vehicle]["target_evader_dis_last"]
                    num_capture = self.pursuit_vehs[vehicle]["num_capture"]
                    self.pursuit_vehs[vehicle] = {"x": p_x,
                                                  "y": p_y,
                                                  "p_lane": p_lane,
                                                  "p_edge": p_lane.split("_")[0],
                                                  "p_lane_left": p_turn_term["l"],
                                                  "p_lane_straight": p_turn_term["s"],
                                                  "p_lane_right": p_turn_term["r"],
                                                  "p_lane_position": p_lane_position,
                                                  "p_target": p_target,
                                                  "target_evader": target_evader,
                                                  "target_evader_dis": target_evader_dis,
                                                  "target_evader_dis_last": target_evader_dis_last,
                                                  "num_capture": num_capture}

            if "e" in vehicle:
                e_x, e_y = traci.vehicle.getPosition(vehicle)
                e_lane = traci.vehicle.getLaneID(vehicle)
                e_lane = self.checkLane(e_lane)
                next_lane_links = traci.lane.getLinks(e_lane)
                e_turn_term = get_turn_lane(next_lane_links)
                e_lane_position = traci.vehicle.getLanePosition(vehicle)
                e_target = traci.vehicle.getRoute(vehicle)[-1]
                self.evader_vehs[vehicle] = {"x": e_x,
                                             "y": e_y,
                                             "e_lane": e_lane,
                                             "e_edge": e_lane.split("_")[0],
                                             "e_lane_left": e_turn_term["l"],
                                             "e_lane_straight": e_turn_term["s"],
                                             "e_lane_right": e_turn_term["r"],
                                             "e_lane_position": e_lane_position,
                                             "e_target": e_target}

        if self.params["no_task"]:
            self.withoutAssignEvader()
        else:
            self.assignEvader()

        if_stop = self.checkPursuit()

        rewards = self.calculateReward()
        eva_rewards = self.calculatedRewardEva(rewards)
        # ========================================判断是否终止=============================================
        if if_stop:
            return True, rewards, eva_rewards
        else:

            # ==============================统计车流信息，更新背景车辆路径=====================================
            if len(self.vehicles) > 0:
                # =============================统计每条车道上的车辆数目=======================================
                for lane_i in range(len(self.laneList)):
                    self.lane_vehs[self.laneList[lane_i]] = 0

                for id_num in range(len(self.vehicles)):
                    # ===============================为背景车辆重新规划路径===================================
                    if "Background" in self.vehicles[id_num]:
                        current_edge = traci.vehicle.getLaneID(self.vehicles[id_num]).split("_")[0]
                        route_last_edge = traci.vehicle.getRoute(self.vehicles[id_num])[-1]
                        if current_edge == route_last_edge:
                            next_edges = self.topology.out_edges(current_edge)
                            next_edge_target = random_select_next_lane(next_edges)
                            route_list = list(next_edge_target)
                            traci.vehicle.setRoute(self.vehicles[id_num], route_list)

                    # =================================计算车流量===========================================
                    current_lane = traci.vehicle.getLaneID(self.vehicles[id_num])
                    if current_lane in self.laneList:
                        self.lane_vehs[current_lane] += 1
                    else:
                        self.lane_vehs[self.junctionLinks[current_lane]] += 1
            self.generateState()
            self.generateOPState()

            self.observation_state.append(self.observation_state_)
            self.nr_history += 1
            if self.nr_history > self.params["history_length"]:
                self.observation_state.pop(0)
                self.nr_history -= 1


            # self.pursuitVehControl(choice_random=True)
            # self.evadeVehControl(choice_random=True)

            return False, rewards, eva_rewards

    # ==========================================检查逃避车辆是否被追到======================================
    def checkPursuit(self):
        remove_list = []
        for evader_id in self.evader_vehs.keys():
            e_x, e_y = self.evader_vehs[evader_id]["x"], self.evader_vehs[evader_id]["y"]
            for pursuit_id in self.pursuit_vehs.keys():
                p_x, p_y = self.pursuit_vehs[pursuit_id]["x"], self.pursuit_vehs[pursuit_id]["y"]
                dis_p_e = utils.calculate_dis(e_x, e_y, p_x, p_y)
                # Change the capture distance form 5 to 8
                if dis_p_e < 8:
                    if evader_id not in remove_list:
                        traci.vehicle.remove(evader_id)
                        remove_list.append(evader_id)
                    else:
                        print("%s had been removed!" % evader_id)

                    self.pursuit_vehs[pursuit_id]["num_capture"] += 1
        if len(remove_list) > 0:
            for rm_id in remove_list:
                print("remove: %s" % rm_id)
                try:
                    if rm_id in self.vehicle_list:
                        del self.vehicle_list[rm_id]
                    else:
                        print("%s had been removed!" % rm_id)
                    if rm_id in self.evader_vehs:
                        del self.evader_vehs[rm_id]
                    else:
                        print("%s had been removed!" % rm_id)
                except:
                    pass
                finally:
                    pass
            self.vehicles = traci.vehicle.getIDList()
            # ======================================判断终止条件========================================
            if len(self.evader_vehs) == 0:
                return True
        return False

    def generateOPState(self):
        if self.params["global_ob"]:
            self.op_observation_state = []
            for evader_id in list(self.evader_vehs.keys()):
                veh_lane_code = [0] * len(self.lane2num.keys())
                # eva_lane = self.checkLane(traci.vehicle.getLaneID(evader_id)).split("_")[0]
                # veh_lane_code[self.lane2num[eva_lane]] = 1
                for pursuit_id in self.pursuit_vehs.keys():
                    pur_current_lane = self.checkLane(traci.vehicle.getLaneID(pursuit_id)).split("_")[0]
                    veh_lane_code[self.lane2num[pur_current_lane]] = 1
                self.op_observation_state.append(veh_lane_code)
        else:
            self.op_observation_state = []
            for evader_id in list(self.evader_vehs.keys()):
                all_check_lanes = []
                current_lane = self.checkLane(traci.vehicle.getLaneID(evader_id))
                all_check_lanes.append(current_lane)
                eva_state = [evader_id]
                for command in [0, 1, 2]:  # ["l", "s", "r"]
                    action_next_lane, action_true = get_action(current_lane, command)
                    if action_true:
                        all_check_lanes.append(action_next_lane)
                    else:
                        all_check_lanes.append(None)
                for check_lane in all_check_lanes:
                    if check_lane is not None:
                        eva_state += [0, 0]
                        for pursuit_id in self.pursuit_vehs.keys():
                            pur_current_lane = self.checkLane(traci.vehicle.getLaneID(pursuit_id))
                            if pur_current_lane == check_lane:
                                if traci.vehicle.getLanePosition(pursuit_id) <= 50:
                                    eva_state[-2] = 1
                                    break
                                else:
                                    eva_state[-1] = 1
                                    break
                    else:
                        eva_state += [0, 0]

                self.op_observation_state.append(eva_state)

    def generateState(self):
        self.observation_state_ = []
        self.global_state = []

        for pursuit_id in list(self.pursuit_vehs.keys()):
            p_edge = self.lane2num[self.pursuit_vehs[pursuit_id]["p_edge"]]
            p_length = self.pursuit_vehs[pursuit_id]["p_lane_position"]/self.params["lane_length"]
            p_edge_code = get_bin(p_edge, self.params["code_length"])
            pur_state = [pursuit_id]
            pur_state += p_edge_code
            pur_state.append(p_length)
            for turn_term in ["p_lane_left", "p_lane_straight", "p_lane_right"]:
                if self.pursuit_vehs[pursuit_id][turn_term] is not None:
                    pur_state += get_bin(self.lane2num[self.pursuit_vehs[pursuit_id][turn_term]], self.params["code_length"])
                else:
                    pur_state += [0 for _ in range(self.params["code_length"])]

            #====================观测内的evader======================================
            all_check_lanes = []
            current_lane = self.checkLane(traci.vehicle.getLaneID(pursuit_id))
            all_check_lanes.append(current_lane)
            for command in [0, 1, 2]:  # ["l", "s", "r"]
                action_next_lane, action_true = get_action(current_lane, command)
                if action_true:
                    all_check_lanes.append(action_next_lane)
                else:
                    all_check_lanes.append(None)

            for check_lane in all_check_lanes:
                if check_lane is not None:
                    # 每一条道路划分两个格子
                    pur_state += [0, 0]
                    for evader_id in self.evader_vehs.keys():
                        eva_current_lane = self.checkLane(traci.vehicle.getLaneID(evader_id))
                        # 如果pursuer观测内的话
                        if eva_current_lane == check_lane:
                            # 将道路划分成了两个格子
                            if traci.vehicle.getLanePosition(evader_id) <= 50:
                                # 道路一共有100米，第一个50米，记录在第一个格子里
                                pur_state[-2] = 1
                                break
                            else:
                                pur_state[-1] = 1
                                break
                else:
                    pur_state += [0, 0]

            self.observation_state_.append(pur_state)
            self.global_state.append(pur_state)

        for add_i in range(self.params["num_pursuit"] - len(self.pursuit_vehs.keys())):
            self.observation_state_.append([0 for _ in range(self.params["local_observation_shape"][1])])
            self.global_state.append([0 for _ in range(self.params["global_observation_shape"][1])])

        # for evader_id in list(self.evader_vehs.keys()):
        #     e_edge = self.lane2num[self.evader_vehs[evader_id]["e_edge"]]
        #     e_length = self.evader_vehs[evader_id]["e_lane_position"]/self.params["lane_length"]
        #     e_edge_code = get_bin(e_edge, self.params["code_length"])
        #     eva_state = [evader_id]
        #     eva_state += e_edge_code
        #     eva_state.append(e_length)
        #     eva_state_global = copy.deepcopy(eva_state)
        #     route_last_edge = traci.vehicle.getRoute(evader_id)[-1]
        #     for turn_term in ["e_lane_left", "e_lane_straight", "e_lane_right"]:
        #         if self.evader_vehs[evader_id][turn_term] is not None:
        #             eva_state += get_bin(self.lane2num[self.evader_vehs[evader_id][turn_term]], self.params["code_length"])
        #             if route_last_edge == self.evader_vehs[evader_id][turn_term]:
        #                 eva_state_global += get_bin(self.lane2num[route_last_edge], self.params["code_length"])
        #             else:
        #                 eva_state_global += [0 for _ in range(self.params["code_length"])]
        #         else:
        #             eva_state += [0 for _ in range(self.params["code_length"])]
        #             eva_state_global += [0 for _ in range(self.params["code_length"])]
        #
        #     # eva_state += vehNumPre
        #     # eva_state_global += vehNumPre
        #
        #     # self.observation_state.append(eva_state)
        #     self.global_state.append(eva_state_global)

        # for add_i in range(self.params["num_evader"] - len(self.evader_vehs.keys())):
        #     self.observation_state.append([0 for _ in range(self.params["local_observation_shape"][1])])
        #     self.global_state.append([0 for _ in range(self.params["global_observation_shape"][1])])

    def pursuitVehControl(self, choice_random=False, commands=None):
        if choice_random:
            for pursuit_id in self.pursuit_vehs.keys():
                # ===============================追逐车辆随机选择目的地===================================
                current_edge = traci.vehicle.getLaneID(pursuit_id).split("_")[0]
                route_last_edge = traci.vehicle.getRoute(pursuit_id)[-1]
                if current_edge == route_last_edge:
                    next_edges = self.topology.out_edges(current_edge)
                    next_edge_target = random_select_next_lane(next_edges)
                    route_list = list(next_edge_target)
                    traci.vehicle.setRoute(pursuit_id, route_list)
        else:
            assert commands.shape == (self.params["num_pursuit"], )
            for _i, pur_veh in enumerate(self.params["pursuit_ids"]):
                if pur_veh in self.pursuit_vehs.keys():
                    current_lane = self.checkLane(traci.vehicle.getLaneID(pur_veh))
                    action_next_lane, action_true = get_action(current_lane, commands[_i])
                    route_list = [current_lane.split("_")[0], action_next_lane]
                    traci.vehicle.setRoute(pur_veh, route_list)
                else:
                    continue

    def evadeVehControl(self, choice="random_choice", commands=None):
        if choice == "random_choice":
            for evader_id in self.evader_vehs.keys():
                # ===============================追逐车辆随机选择目的地===================================
                current_edge = traci.vehicle.getLaneID(evader_id).split("_")[0]
                route_last_edge = traci.vehicle.getRoute(evader_id)[-1]
                if current_edge == route_last_edge:
                    next_edges = self.topology.out_edges(current_edge)
                    next_edge_target = random_select_next_lane(next_edges)
                    route_list = list(next_edge_target)
                    traci.vehicle.setRoute(evader_id, route_list)
        elif choice == "Stop":
            for evader_id in self.evader_vehs.keys():
                current_edge = traci.vehicle.getLaneID(evader_id).split("_")[0]
                current_pos = traci.vehicle.getLanePosition(evader_id)
                traci.vehicle.setStop(evader_id, current_edge, current_pos, 0)
        elif choice == "Command":
            assert len(commands) == len(self.evader_vehs.keys())
            for _i, eva_veh in enumerate(self.evader_vehs.keys()):
                if eva_veh in self.evader_vehs.keys():
                    current_lane = self.checkLane(traci.vehicle.getLaneID(eva_veh))
                    action_next_lane, action_true = get_action(current_lane, commands[_i])
                    route_list = [current_lane.split("_")[0], action_next_lane]
                    traci.vehicle.setRoute(eva_veh, route_list)
                else:
                    continue

    def checkLane(self, lane):
        if "J" in lane:
            next_lane = self.junctionLinks[lane]
            # route_list = traci.vehicle.getRoute(vehicle_id)
            return next_lane
        else:
            return lane

    def calculateReward(self):
        inter_dis = 10
        rewards = []
        # 时间步损失
        for pursuit_id in self.pursuit_vehs.keys():
            reward = -1
            reward += self.pursuit_vehs[pursuit_id]["num_capture"]*10
            self.pursuit_vehs[pursuit_id]["num_capture"] = 0
            # reward += (self.pursuit_vehs[pursuit_id]["target_evader_dis_last"] - self.pursuit_vehs[pursuit_id]["target_evader_dis"])
            # reward += (inter_dis - self.pursuit_vehs[pursuit_id]["target_evader_dis"])/(inter_dis/2)
            rewards.append(reward)
        return rewards

    def calculatedRewardEva(self, pursuit_rewards):
        inter_dis = 10
        rewards = []
        for evader_id in self.params["evader_ids"]:
            if evader_id in self.evader_vehs.keys():
                for i, pursuit_id in enumerate(self.pursuit_vehs.keys()):
                    if self.pursuit_vehs[pursuit_id]["target_evader"] == evader_id:
                        reward = -1 * pursuit_rewards[i]
                        rewards.append(reward)
                        break
        if len(rewards) == 1:
            a = 1
        return rewards

    def withoutAssignEvader(self):
        for pur_index, pursuit_id in enumerate(self.pursuit_vehs.keys()):
            dis_list = []
            for eva_index, evader_id in enumerate(self.evader_vehs.keys()):
                eva_x, eva_y = self.evader_vehs[evader_id]["x"], self.evader_vehs[evader_id]["y"]
                dis = utils.calculate_dis(eva_x, eva_y,
                                          self.pursuit_vehs[pursuit_id]["x"], self.pursuit_vehs[pursuit_id]["y"])
                dis_list.append(dis)
            self.pursuit_vehs[pursuit_id]["target_evader_dis_last"] = \
                self.pursuit_vehs[pursuit_id]["target_evader_dis"]
            self.pursuit_vehs[pursuit_id]["target_evader_dis"] = np.mean(dis_list)

    def assignEvader(self):
        num_pairs = len(self.evader_vehs.keys())
        if num_pairs > 0:
            purs_evas_dis_dict = []
            purs_evas_dis_dict_total = []

            for eva_index, evader_id in enumerate(self.evader_vehs.keys()):
                purs_eva_dis_dict = []
                eva_x, eva_y = self.evader_vehs[evader_id]["x"], self.evader_vehs[evader_id]["y"]
                for pur_index, pursuit_id in enumerate(self.pursuit_vehs.keys()):
                    dis = utils.calculate_dis(eva_x, eva_y,
                                              self.pursuit_vehs[pursuit_id]["x"], self.pursuit_vehs[pursuit_id]["y"])
                    purs_eva_dis_dict.append({"pursuit": pursuit_id,
                                              "evader": evader_id,
                                              "dis": dis})
                    purs_evas_dis_dict_total.append({"pursuit": pursuit_id,
                                                     "evader": evader_id,
                                                     "dis": dis})
                purs_evas_dis_dict.append(purs_eva_dis_dict)

            if self.params["assign_method"] == "greedy":
                smallest_dis = heapq.nsmallest(int(len(self.pursuit_vehs.keys()) / len(self.evader_vehs.keys())),
                                               purs_evas_dis_dict[0], lambda x: x["dis"])
                for term in smallest_dis:
                    self.pursuit_vehs[term["pursuit"]]["target_evader"] = term["evader"]
                    self.pursuit_vehs[term["pursuit"]]["target_evader_dis_last"] = \
                        self.pursuit_vehs[term["pursuit"]]["target_evader_dis"]
                    self.pursuit_vehs[term["pursuit"]]["target_evader_dis"] = term["dis"]

                if num_pairs > 1:
                    other_eva = purs_evas_dis_dict[1][0]["evader"]
                    for pursuit_id in self.pursuit_vehs.keys():
                        if self.pursuit_vehs[pursuit_id]["target_evader"] is None:
                            self.pursuit_vehs[pursuit_id]["target_evader"] = other_eva
                            self.pursuit_vehs[pursuit_id]["target_evader_dis_last"] =\
                                self.pursuit_vehs[pursuit_id]["target_evader_dis"]
                            self.pursuit_vehs[pursuit_id]["target_evader_dis"] = utils.calculate_dis(
                                self.pursuit_vehs[pursuit_id]["x"], self.pursuit_vehs[pursuit_id]["y"],
                                self.evader_vehs[other_eva]["x"], self.evader_vehs[other_eva]["y"])
            elif self.params["assign_method"] == "task_allocation":
                label_of_each_evader = {}
                label_of_each_pursuit = {}
                for eva_id in self.evader_vehs.keys():
                    label_of_each_evader[eva_id] = False
                for pur_id in self.pursuit_vehs.keys():
                    label_of_each_pursuit[pur_id] = False
                purs_evas_dis_dict_total.sort(key=lambda x: x["dis"])
                for temp_index in range(len(purs_evas_dis_dict_total)):
                    if False not in label_of_each_evader.values() and False not in label_of_each_pursuit.values():
                        break
                    evader_id_temp = purs_evas_dis_dict_total[temp_index]["evader"]
                    pursuit_id_temp = purs_evas_dis_dict_total[temp_index]["pursuit"]
                    if self.pursuit_vehs[pursuit_id_temp]["target_evader"] is None:
                        self.pursuit_vehs[pursuit_id_temp]["target_evader"] = evader_id_temp
                        self.pursuit_vehs[pursuit_id_temp]["target_evader_dis_last"] = \
                            self.pursuit_vehs[pursuit_id_temp]["target_evader_dis"]
                        self.pursuit_vehs[pursuit_id_temp]["target_evader_dis"] = \
                            purs_evas_dis_dict_total[temp_index]["dis"]
                        label_of_each_evader[evader_id_temp] = True
                        label_of_each_pursuit[pursuit_id_temp] = True
                    else:
                        continue

    def initPre(self):
        # for i in range(19):
        #     his = [0 for _ in range(48)]
        #     self.pr.save_his(his)
        if self.params["pre_method"] == "informer":
            for i in range(84):
                his = [0 for _ in range(48)]
                self.pr.save_his(his)
        else:
            for i in range(84):
                self.pr.store_his([0 for _ in range(48)])

    def vehFlowPredict(self, now_flow, preFormat=0):
        if self.params["pre_method"] == "informer":

            self.pr.save_his(now_flow)
            # assert preFormat in [0, 1, 2, 3, 4], "The format of predicting is error!"
            return self.pr.pre()[0][-1]
        else:
            self.pr.store_his(now_flow)
            return self.pr.predict()[-1]

        # if preFormat == 0:
        #     return self.pr.pre()[0][0]
        # elif preFormat == 1:
        #     return self.pr.pre()[0][1]
        # elif preFormat == 2:
        #     return self.pr.pre()[0][2]
        # elif preFormat == 3:
        #     return self.pr.pre()[0][3]
        # elif preFormat == 4:
        #     return self.pr.pre()[0][4]

    def initState_his(self):
        state_histories = []
        for i in range(self.params["global_observation_shape"][0]):
            temp_history = []
            for j in range(self.params["global_observation_shape"][1]):
                temp_history.append([0 for _ in range(self.params["global_observation_shape"][2])])
                # state_histories.append([0 for _ in range(self.params["global_observation_shape"][2])])
            state_histories.append(temp_history)
        return state_histories





