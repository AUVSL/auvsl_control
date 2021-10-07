import numpy as np
import math

def angdiff(th1, th2):
    d = th1 - th2
    d = np.mod(d+np.pi, 2*np.pi) - np.pi
    return -d
    
def wraptopi(x):
    pi = np.pi
    x = x - np.floor(x/(2*pi)) *2 *pi
    if (x >= pi):
        return x -2*pi
    return x

def reward(errors, linear_vel, angular_vel):
    DE_penalty_gain = 25
    DE_penalty_shape = 1
    HE_penalty_gain = 25
    HE_penalty_shape = 3
    HE_iwrt_DE = 2
    TDD_reward_gain = 5
    TDD_iwrt_DE = 5
    vel_reward_gain = 1
    vel_iwrt_DE = 1
    steering_penalty_gain = 1
    steering_iwrt_DE = 4

    dis = errors[0]
    theta_far = errors[1]
    theta_near = errors[2]

    dis_temp = np.abs(dis)/1.0
    dis = (math.pow(dis_temp,DE_penalty_shape) + dis_temp) * -DE_penalty_gain

    theta_near_temp = theta_near / np.pi
    theta_near = math.pow(theta_near_temp,HE_penalty_shape) * HE_penalty_gain / (np.exp(dis_temp*HE_iwrt_DE)) * -15

    theta_far_temp = np.abs(theta_far) / np.pi
    theta_far = math.pow(theta_far_temp, HE_penalty_shape) * HE_penalty_gain / (np.exp(dis_temp*HE_iwrt_DE)) * -1.5

    linear_vel = linear_vel * vel_reward_gain / (np.exp(dis_temp* vel_iwrt_DE))

    angular_vel = np.abs(angular_vel) * steering_penalty_gain / (np.exp(dis_temp * steering_iwrt_DE)) * -1

    rewards = dis + theta_near + theta_far + linear_vel + angular_vel
    return rewards
