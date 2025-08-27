#!/usr/bin/python


import matplotlib.pyplot as plt
import numpy as np
import torch
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def getReward3(iifds, actions, obs, global_reward, all_opp):
    num_agent = int(iifds.numberofcar / 2)
    rewards = np.zeros(num_agent)
    correct_action = 1
    wrong_action = -1
    # actions: 0:搜索, 1:逃跑, 2:追击, 3:支援
    for car_id in range(num_agent):
        situation_enemy = obs[car_id][5] * 3 + obs[car_id][7]  # 敌方态势得分（有利态势追击，反之逃跑）
        situation_nei = obs[car_id][1] * 3 + obs[car_id][3]  # 友方态势得分（相对朝向友方时支援，反之搜索）

        if all_opp[car_id] != []:
            has_enemy = True
        else:
            has_enemy = False

        if has_enemy:
            # 存在敌军时策略
            if situation_enemy >= 0:
                if actions[car_id] == 2:  # 正确执行追击
                    rewards[car_id] += correct_action
                else:  # 错误动作
                    rewards[car_id] += wrong_action
            else:
                if actions[car_id] == 1:  # 正确执行逃跑
                    rewards[car_id] += correct_action
                else:  # 错误动作
                    rewards[car_id] += wrong_action
        else:
            # 没有敌军时策略
            if situation_nei >= 0:  # 这里包含了死亡的情况
                if actions[car_id] == 0:  # 正确执行搜索
                    rewards[car_id] += correct_action
                else:  # 错误动作
                    rewards[car_id] += wrong_action
            else:
                if actions[car_id] == 3:  # 正确执行支援
                    rewards[car_id] += correct_action
                else:  # 错误动作
                    rewards[car_id] += wrong_action
        rewards[car_id] += global_reward

    return rewards


def getReward2(iifds, actions, obs, global_reward, all_opp, all_nei_c2e, obs_dim):
    num_agent = int(iifds.numberofcar / 2)
    rewards = np.zeros(num_agent)
    correct_action = 1
    wrong_action = -1
    # actions: 0:搜索, 1:逃跑, 2:追击, 3:支援
    for car_id in range(num_agent):
        situation_enemy = obs[car_id][int(2 * obs_dim / 4 + 1)] * 3 + obs[car_id][
            int(3 * obs_dim / 4 + 1)]  # 敌方态势得分（有利态势追击，反之逃跑）
        situation_nei = obs[car_id][int(0 * obs_dim / 4 + 1)] * 3 + obs[car_id][
            int(1 * obs_dim / 4 + 1)]  # 友方态势得分（相对朝向友方时支援，反之搜索）

        if all_opp[car_id+num_agent]:
            has_enemy = True
        else:
            has_enemy = False

        if all_nei_c2e[car_id+num_agent]:
            has_ally = True
        else:
            has_ally = False

        if has_enemy:
            # 存在敌军时策略
            if situation_enemy >= 0:
                if actions[car_id] == 2:  # 正确执行追击
                    rewards[car_id] += correct_action
                else:  # 错误动作
                    rewards[car_id] += wrong_action
            else:
                if actions[car_id] == 1:  # 正确执行逃跑
                    rewards[car_id] += correct_action
                else:  # 错误动作
                    rewards[car_id] += wrong_action
        else:
            # 没有敌军时策略
            if situation_nei >= 0 and has_ally:  # 这里包含了死亡的情况
                if actions[car_id] == 3:  # 正确执行支援
                    rewards[car_id] += correct_action
                else:  # 错误动作
                    rewards[car_id] += wrong_action
            else:
                if actions[car_id] == 0:  # 正确执行搜索
                    rewards[car_id] += correct_action
                else:  # 错误动作
                    rewards[car_id] += wrong_action
        rewards[car_id] += global_reward

    return rewards


def getReward1(iifds, actions, obs, global_reward, all_opp, all_nei_c2e, obs_dim, rewards):
    num_agent = int(iifds.numberofcar / 2)
    correct_action = 1
    wrong_action = -1
    rewards_loss = np.zeros(num_agent)
    # actions: 0:搜索, 1:逃跑, 2:追击, 3:支援
    for car_id in range(num_agent):
        situation_enemy = obs[car_id][int(2*obs_dim/4+1)] * 3 + obs[car_id][int(3*obs_dim/4+1)]  # 敌方态势得分（有利态势追击，反之逃跑）
        situation_nei = obs[car_id][int(0*obs_dim/4+1)] * 3 + obs[car_id][int(1*obs_dim/4+1)]  # 友方态势得分（相对朝向友方时支援，反之搜索）

        if all_opp[car_id]:
            has_enemy = True
        else:
            has_enemy = False

        if all_nei_c2e[car_id]:
            has_ally = True
        else:
            has_ally = False

        if has_enemy:
            # 存在敌军时策略
            if situation_enemy >= 0:
                if actions[car_id] == 2:  # 正确执行追击
                    rewards[car_id] += correct_action
                    rewards_loss[car_id] += correct_action
                else:  # 错误动作
                    rewards[car_id] += wrong_action
                    rewards_loss[car_id] += wrong_action
            else:
                if actions[car_id] == 1:  # 正确执行逃跑
                    rewards[car_id] += correct_action
                    rewards_loss[car_id] += correct_action
                else:  # 错误动作
                    rewards[car_id] += wrong_action
                    rewards_loss[car_id] += wrong_action
        else:
            # 没有敌军时策略
            if situation_nei >= 0 and has_ally:  # 这里包含了死亡的情况
                if actions[car_id] == 3:  # 正确执行支援
                    rewards[car_id] += correct_action
                    rewards_loss[car_id] += correct_action
                else:  # 错误动作
                    rewards[car_id] += wrong_action
                    rewards_loss[car_id] += wrong_action
            else:
                if actions[car_id] == 0:  # 正确执行搜索
                    rewards[car_id] += correct_action
                    rewards_loss[car_id] += correct_action
                else:  # 错误动作
                    rewards[car_id] += wrong_action
                    rewards_loss[car_id] += wrong_action
        rewards[car_id] += global_reward
        rewards_loss[car_id] += global_reward
    return rewards, rewards_loss


def get_reward_multiple(env, qNext, dic):
    """多动态障碍环境获取reward函数"""
    reward = 0
    distance = env.distanceCost(qNext, dic['obsCenter'])
    if distance <= dic['obs_r']:
        reward += (distance - dic['obs_r']) / dic['obs_r'] - 1
    else:
        if distance < dic['obs_r'] + 0.4:
            tempR = dic['obs_r'] + 0.4
            reward += (distance - tempR) / tempR - 0.3
        distance1 = env.distanceCost(qNext, env.goal)
        distance2 = env.distanceCost(env.start, env.goal)
        if distance1 > env.threshold:
            reward += -distance1 / distance2
        else:
            reward += -distance1 / distance2 + 3
    return reward


def drawActionCurve(actionCurveList):
    """
    :param actionCurveList: 动作值列表
    :return: None 绘制图像
    """
    plt.figure()
    for i in range(actionCurveList.shape[1]):
        array = actionCurveList[:, i]
        if i == 0: label = 'row01'
        if i == 1: label = 'sigma01'
        if i == 2: label = 'theta1'
        if i == 3: label = 'row02'
        if i == 4: label = 'sigma02'
        if i == 5: label = 'theta2'
        plt.plot(np.arange(array.shape[0]), array, linewidth=2, label=label)
    plt.title('Variation diagram')
    plt.grid()
    plt.xlabel('time')
    plt.ylabel('value')
    plt.legend(loc='best')


def checkPath(apf):
    sum = 0
    for i in range(apf.path.shape[0] - 1):
        sum += apf.distanceCost(apf.path[i, :], apf.path[i + 1, :])
    for i, j in zip(apf.path, apf.dynamicSphere_Path):
        if apf.distanceCost(i, j) <= apf.dynamicSphere_R:
            print('与障碍物有交点，轨迹距离为：', sum)
            return
    print('与障碍物无交点，轨迹距离为：', sum)


def transformAction(actionBefore, actionBound, actionDim):
    """将强化学习输出的动作映射到指定的动作范围"""
    actionAfter = []
    for i in range(actionDim):
        action_i = actionBefore[i]
        action_bound_i = actionBound[i]
        actionAfter.append((action_i + 1) / 2 * (action_bound_i[1] - action_bound_i[0]) + action_bound_i[0])
    return actionAfter


def test(iifds, actors_cur, dynamicController, arglist, conf):
    """动态单障碍环境测试训练效果"""

    q = []
    qBefore = []
    v = []
    vObs = []
    start = iifds.start
    goal = iifds.goal

    obsCenter = [np.array([-8.5, -1.5, 0.8]), np.array([-9, -1.5, 0.8]),
                 np.array([-9.5, -1.5, 0.8]), np.array([-10, -1.5, 0.8]), np.array([-4, -1.5, 0.8]),

                 np.array([-4.5, -1.5, 0.8]), np.array([-5, -1.5, 0.8]), np.array([-5.5, -1.5, 0.8]),
                 np.array([-6, -1.5, 0.8]), np.array([-6.5, -1.5, 0.8]), np.array([-7, -1.5, 0.8]),
                 np.array([-7.5, -1.5, 0.8]), np.array([-8, -1.5, 0.8]),  # 序号1

                 np.array([-8.5, -4.5, 0.8]), np.array([-9, -4.5, 0.8]),
                 np.array([-9.5, -4.5, 0.8]), np.array([-10, -4.5, 0.8]), np.array([-4, -4.5, 0.8]),

                 np.array([-4.5, -4.5, 0.8]), np.array([-5, -4.5, 0.8]), np.array([-5.5, -4.5, 0.8]),
                 np.array([-6, -4.5, 0.8]), np.array([-6.5, -4.5, 0.8]), np.array([-7, -4.5, 0.8]),
                 np.array([-7.5, -4.5, 0.8]), np.array([-8, -4.5, 0.8]),  # 序号2

                 np.array([-13.5, 4, 0.8]), np.array([-13.5, 3.5, 0.8]), np.array([-13.5, 3, 0.8]),
                 np.array([-13.5, 2.5, 0.8]), np.array([-13.5, 2, 0.8]), np.array([-13.5, 1.5, 0.8]),  # 序号4

                 np.array([-11.5, -6.5, 0.8]), np.array([-12, -6.5, 0.8]), np.array([-12.5, -6.5, 0.8]),
                 np.array([-13, -6.5, 0.8]), np.array([-13.5, -6.5, 0.8]), np.array([-14, -6.5, 0.8]),  # 序号5

                 np.array([-7.5, 6.5, 0.8]), np.array([-7.5, 7, 0.8]), np.array([-7.5, 7.5, 0.8]),
                 np.array([-7.5, 8, 0.8]),  # 序号6

                 # 序号6#

                 # np.array([7.5, 4, 0.8]), np.array([7.5, 4.5, 0.8]), np.array([7.5, 5, 0.8]),
                 # np.array([7.5, 5.5, 0.8]), np.array([7.5, 6, 0.8]), np.array([7.5, 6.5, 0.8]),  #   序号12

                 np.array([9.5, 6, 0.8]), np.array([10, 6, 0.8]), np.array([10.5, 6, 0.8]),
                 np.array([11, 6, 0.8]), np.array([11.5, 6, 0.8]), np.array([12, 6, 0.8]),  # 序号16

                 # np.array([4, 2, 0.8]), np.array([4, 1.5, 0.8]), np.array([4, 1, 0.8]),
                 # np.array([4, 0.5, 0.8]), np.array([4, 0, 0.8]), np.array([4, -0.5, 0.8]),
                 # np.array([4, -1, 0.8]), np.array([4, -1.5, 0.8]), np.array([4, -2, 0.8]),  #序号13

                 np.array([4, -5.5, 0.8]), np.array([4.5, -5.5, 0.8]), np.array([5, -5.5, 0.8]),
                 np.array([5.5, -5.5, 0.8]), np.array([6, -5.5, 0.8]), np.array([6.5, -5.5, 0.8]),
                 np.array([7, -5.5, 0.8]), np.array([7.5, -5.5, 0.8]), np.array([8, -5.5, 0.8]),  # 序号14

                 np.array([10, -2, 0.8]), np.array([10.5, -2, 0.8]), np.array([11, -2, 0.8]),
                 np.array([11.5, -2, 0.8]), np.array([12, -2, 0.8]), np.array([12.5, -2, 0.8]),
                 np.array([13, -2, 0.8]),  # 序号15

                 np.array([-16, 8, 0.8]), np.array([-16, 8.5, 0.8]), np.array([-16, 9, 0.8]),
                 # np.array([-10.5, 8, 0.8]), np.array([-10.5, 7.5, 0.8]), np.array([-10.5, 7, 0.8]), np.array([-10.5, 6.5, 0.8]),  #序号8

                 np.array([6.5, 2, 0.8]), np.array([6.5, 2.5, 0.8]), np.array([6.5, 3, 0.8]),
                 np.array([6.5, 3.5, 0.8]), np.array([6.5, 4, 0.8]), np.array([6.5, 4.5, 0.8]), np.array([6.5, 5, 0.8]),
                 # 序号11

                 np.array([1, 7, 0.8]), np.array([0.5, 7, 0.8]),
                 np.array([0, 7, 0.8]), np.array([-0.5, 7, 0.8]), np.array([-1, 7, 0.8]),

                 np.array([1, 6.5, 0.8]), np.array([1, 6, 0.8]),
                 np.array([-1, 6.5, 0.8]), np.array([-1, 6, 0.8]),  # 序号9

                 np.array([1, -8.5, 0.8]), np.array([0.5, -8.5, 0.8]),
                 np.array([0, -8.5, 0.8]), np.array([-0.5, -8.5, 0.8]), np.array([-1, -8.5, 0.8]),

                 np.array([1, -7.5, 0.8]), np.array([1, -8, 0.8]),
                 np.array([-1, -7.5, 0.8]), np.array([-1, -8, 0.8]),  # 序号10

                 np.array([-5.5, 3.5, 0.8]), np.array([-6, 3.5, 0.8]),
                 np.array([-6.5, 3.5, 0.8]), np.array([-7, 3.5, 0.8]), np.array([-7.5, 3.5, 0.8]),

                 np.array([-5.5, 3, 0.8]), np.array([-5.5, 2.5, 0.8]),
                 np.array([-7.5, 3, 0.8]), np.array([-7.5, 2.5, 0.8]),  # 序号7

                 # np.array([-14, 3, 0.8]), np.array([-14, 2.5, 0.8]), np.array([-14, 2, 0.8]),
                 # np.array([-14, 1.5, 0.8]), np.array([-14, 1, 0.8]), np.array([-14, 0.5, 0.8]),
                 # np.array([-14, 0, 0.8]),
                 # np.array([-13.5, 2.5, 0.8]), np.array([-13, 2.5, 0.8]), np.array([-12.5, 2.5, 0.8]),
                 # np.array([-13.5, 0.5, 0.8]), np.array([-13, 0.5, 0.8]), np.array([-12.5, 0.5, 0.8]),  #序号3

                 np.array([13.5, 3.5, 0.8]), np.array([13.5, 3, 0.8]),
                 np.array([13.5, 2.5, 0.8]), np.array([13.5, 2, 0.8]), np.array([13.5, 1.5, 0.8]),

                 np.array([13, 3.5, 0.8]), np.array([12.5, 3.5, 0.8]),
                 np.array([13, 1.5, 0.8]), np.array([12.5, 1.5, 0.8]),  # 序号17

                 np.array([14, -5.5, 0.8]), np.array([14, -6, 0.8]),
                 np.array([14, -6.5, 0.8]), np.array([14, -7, 0.8]), np.array([14, -7.5, 0.8]),

                 # np.array([13.5, -5.5, 0.8]), np.array([13, -5.5, 0.8]), np.array([12.5, -5.5, 0.8]),
                 # np.array([13.5, -7.5, 0.8]), np.array([13, -7.5, 0.8]), np.array([12.5, -7.5, 0.8]), # 序号18

                 np.array([0, 0, 0.8])

                 ]
    obs_num = len(obsCenter)
    for i in range(obs_num):
        # obsCenter.append(np.array([obs_init[0][i], obs_init[1][i], obs_init[2][i]]))
        vObs.append(np.array([0, 0, 0], dtype=float))
    iifds.reset(obsCenter)

    for i in range(iifds.numberofcar):
        q.append(start[i])
        qBefore.append([None, None, None])
        v.append((q[i] - q[i]) / iifds.timeStep)
    rewardSum1 = 0
    rewardSum2 = 0
    ta_index = np.ones(iifds.numberofcar) * -3
    dead_index = np.zeros(iifds.numberofcar)
    # assign_index = assign_index.reshape(1, -1)
    ta_index = ta_index.reshape(1, -1)
    dead_index = dead_index.reshape(1, -1)
    flag_car = np.zeros(iifds.numberofcar)
    for i in range(500):
        goal, ass_index, task_index = iifds.assign(q, v, goal, flag_car, -1)

        obsCenterNext = obsCenter
        vObsNext = vObs

        obsDicq = iifds.calDynamicState(q, v, obsCenter, vObs, obs_num, goal, flag_car)  # 相对位置字典
        (obs_n_car1, obs_n_car2, obs_n_car3, obs_n_car4, obs_n_car5, obs_n_car6, obs_n_car7, obs_n_car8, obs_n_car9,
         obs_n_car10, obs_n_car11, obs_n_car12, obs_n_car13, obs_n_car14, obs_n_car15, obs_n_car16, obs_n_car17,
         obs_n_car18, obs_n_car19, obs_n_car20) = obsDicq['car1'], obsDicq['car2'], obsDicq['car3'], obsDicq[
            'car4'], obsDicq['car5'], obsDicq['car6'], obsDicq['car7'], obsDicq['car8'], obsDicq['car9'], obsDicq[
            'car10'], obsDicq['car11'], obsDicq['car12'], obsDicq['car13'], obsDicq['car14'], obsDicq['car15'], obsDicq[
            'car16'], obsDicq['car17'], obsDicq['car18'], obsDicq['car19'], obsDicq['car20']
        obs_n1 = obs_n_car1 + obs_n_car2 + obs_n_car3 + obs_n_car4 + obs_n_car5 + obs_n_car6 + obs_n_car7 + obs_n_car8 + obs_n_car9 + obs_n_car10
        obs_n2 = obs_n_car11 + obs_n_car12 + obs_n_car13 + obs_n_car14 + obs_n_car15 + obs_n_car16 + obs_n_car17 + obs_n_car18 + obs_n_car19 + obs_n_car20

        action_n2 = []
        qNext = []
        vNext = []

        action_n1 = [agent(torch.from_numpy(obs).to(arglist.device, torch.float)).detach().cpu().numpy() \
                     for agent, obs in zip(actors_cur, obs_n1)]
        action_n1 = np.clip(action_n1, arglist.action_limit_min, arglist.action_limit_max)
        action_n1 = action_n1.reshape(-1)

        for j in range(int(iifds.numberofcar / 2)):
            obs_n2_ = torch.as_tensor(obs_n2[j], dtype=torch.float, device=device)
            action_n2_ = dynamicController(obs_n2_).cpu().detach().numpy()
            action_n2.append(transformAction(action_n2_, conf.actionBound, conf.act_dim))

            qNext.append(iifds.getqNext(j, q, v, obsCenter, vObs, action_n1[3 * j], action_n1[3 * j + 1],
                                        action_n1[3 * j + 2], qBefore, goal))
            vNext.append((qNext[j] - q[j]) / iifds.timeStep)

        for j in range(int(iifds.numberofcar / 2)):
            qNext.append(iifds.getqNext(j + int(iifds.numberofcar / 2), q, v, obsCenter, vObs, action_n2[j][0],
                                        action_n2[j][1],
                                        action_n2[j][2], qBefore, goal))
            vNext.append(
                (qNext[j + int(iifds.numberofcar / 2)] - q[j + int(iifds.numberofcar / 2)]) / iifds.timeStep)

        for j in range(iifds.numberofcar):

            if flag_car[j] == 1:
                qNext[j] = q[j]
                vNext[j] = np.array([0, 0, 0])
            else:
                if ass_index[j] != -1:
                    if iifds.distanceCost(goal[j], q[j]) < iifds.threshold:
                        # finish_car.append(ass_index[j])
                        flag_car[ass_index[j]] = 1
                        qNext[int(ass_index[j])] = q[int(ass_index[j])]
                        vNext[int(ass_index[j])] = np.array([0, 0, 0])
                        goal, ass_index, task_index = iifds.assign(q, v, goal, flag_car, -1)
                else:
                    if iifds.distanceCost(goal[j], q[j]) < iifds.threshold2:
                        goal, ass_index, task_index = iifds.assign(q, v, goal, flag_car, j)

        rew_n1 = getReward1(qNext, obsCenterNext, obs_num, goal, iifds, start)  # 每个agent使用相同的reward
        rew_n2 = getReward2(qNext, obsCenterNext, obs_num, goal, iifds, start)  # 每个agent使用相同的reward
        rewardSum1 += rew_n1
        rewardSum2 += rew_n2

        qBefore = q
        q = qNext
        v = vNext
        obsCenter = obsCenterNext
        vObs = vObsNext
        # for j in range(int(iifds.numberofcar / 2)):
        #     goal[j] = q[iifds.ass[j]]

        # if flag_c == int(iifds.numberofcar / 2):
        if sum(flag_car[0:int(iifds.numberofcar / 2)]) == int(iifds.numberofcar / 2) or sum(
                flag_car[int(iifds.numberofcar / 2):iifds.numberofcar]) == int(iifds.numberofcar / 2):
            break
    return rewardSum1, rewardSum2


def setup_seed(seed):
    """设置随机数种子函数"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
