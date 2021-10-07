#!/usr/bin/python3
import rospy
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from auvsl_control.srv import AnfisControl
import numpy as np
import torch
import time
import matplotlib.pyplot as plt
from numpy.linalg import inv
import math
import torch
import anfis
import os
from utils.utils import reward, angdiff, wraptopi
from utils.path import test_course, test_course2

dtype = torch.float

linear_velocity = 1
x = 0.0
y = 0.0
q1 = 0.0
q2 = 0.0
q3 = 0.0
q4 = 0.0
currentAngle = 0.0
control_law = 0.0

stop = False
path_count = 0
batch_size = 128
done = False

robot_path = []
dis_error = []


def fuzzy_error(curr, tar, future):
    global dis_error

    A = np.array([ [curr[1]-tar[1], tar[0]-curr[0]], [tar[0]-curr[0], tar[1]-curr[1]] ] )
    b = np.array([ [tar[0]*curr[1]-curr[0]*tar[1]], [x*(tar[0]-curr[0]) + y*(tar[1]-curr[1])] ])
    proj = np.matmul(inv(A),b)
    d = ( x-curr[0] )*(tar[1]-curr[1]) - (y-curr[1])*(tar[0]-curr[0])

    if ( d >0):
        side = 1
    elif ( d < 0):
        side = -1
    else:
        side = 0

    distanceLine= np.linalg.norm(np.array([x,y])-proj.T,2)*side                     ##########################check this
    dis_error.append(distanceLine)

    farTarget = np.array( [0.9*proj[0] + 0.1*tar[0], 0.9*proj[1] + 0.1*tar[1]] )
    th1 = math.atan2(farTarget[1]-y, farTarget[0]-x)
    th2 = math.atan2(tar[1]-curr[1], tar[0]-curr[0])
    th3 = math.atan2(future[1]-tar[1], future[0]-tar[0])
    theta_far = th1 - currentAngle
    theta_near = th2 - currentAngle
    theta_far = wraptopi(theta_far)
    theta_near = wraptopi(theta_near)


    return [distanceLine,theta_far,theta_near]

def target_generator(path):
    global path_count
    global stop
    global path_length
    path_length = len(path) - 1
    pos_x = x
    pos_y = y

    current_point = np.array( path[path_count] )
    target = np.array( path[path_count + 1] )

    A = np.array([ [(current_point[1]-target[1]),(target[0]-current_point[0])], [(target[0]-current_point[0]), (target[1]-current_point[1])] ])
    b = np.array([ [(target[0]*current_point[1] - current_point[0]*target[1])], [(pos_x*(target[0]-current_point[0]) + pos_y*(target[1] - current_point[1]))] ])
    proj = np.matmul(inv(A),b)

    current_point = np.array( [ [current_point[0]],[current_point[1]] ] )
    target = np.array( [ [target[0]],[target[1]] ] )
    temp1 = proj-current_point      ####dot product
    temp2 = target - current_point
    projLen = (temp1[0]*temp2[0] + temp1[1]*temp2[1])  / np.linalg.norm(target - current_point,2)**2

    if (projLen > 1):
        path_count += 1

    if (path_count == path_length-1):
        stop = True

    if ( (path_count == (path_length-2)) or (path_count == (path_length -1)) ):
        curr = np.array(path[path_count])
        tar = np.array(path[path_count+1])
        future = np.array(path[path_count+1])
    else:
        curr = np.array(path[path_count])
        tar = np.array(path[path_count+1])
        future = np.array(path[path_count+2])

    return curr, tar, future

def callback_ignore(msg):
    global x
    global y
    global q1
    global q2
    global q3
    global q4
    global currentAngle
    global robot_path
    global stop
    global done
    x = msg.pose.pose.position.x
    y = msg.pose.pose.position.y
    q2 = msg.pose.pose.orientation.x
    q3 = msg.pose.pose.orientation.y
    q4 = msg.pose.pose.orientation.z
    q1 = msg.pose.pose.orientation.w
    currentAngle = math.atan2(2*(q1*q4+q2*q3),1-2*(q3**2+q4**2))

    if stop == False:
        robot_path.append([x,y])


def agent_update(new_state, linear_velocity, control_law, agent, done, batch_size, dis_error):
    rewards = reward(new_state, linear_velocity, control_law)
    ####do this every 0.075 s
    # print("11111111111")
    state = agent.curr_states
    new_state = np.array(new_state)
    agent.curr_states = new_state
    agent.memory.push(state,control_law,rewards,new_state,done)   ########control_law aftergain or before gain?
    if len(agent.memory) > batch_size and dis_error > 0.10:
        agent.update(batch_size)


def handleControl(req):
    global x
    global y
    global q1
    global q2
    global q3
    global q4
    global currentAngle
    global robot_path
    
    x = req.pose.position.x
    y = req.pose.position.y
    q2 = req.pose.orientation.x
    q3 = req.pose.orientation.y
    q4 = req.pose.orientation.z
    q1 = req.pose.orientation.w
    currentAngle = math.atan2(2*(q1*q4+q2*q3),1-2*(q3**2+q4**2))

    
    current_point = [req.current_x, req.current_y]
    target_point = [req.target_x, req.target_y]
    future_point = [req.future_x, req.future_y]
    
    new_state = fuzzy_error(current_point, target_point, future_point)
    control_law = agent.get_action(np.array(new_state))
    control_law = control_law.item()
    
    if (control_law > 4.):
        control_law = 4.
    if (control_law < -4.):
        control_law = -4.
    
    twist_msg = Twist()
    twist_msg.linear.x = linear_velocity
    twist_msg.linear.y = 0
    twist_msg.linear.z = 0
    twist_msg.angular.x = 0
    twist_msg.angular.y = 0
    twist_msg.angular.z = control_law
    return twist_msg

if __name__ == "__main__":
    agent = torch.load('/home/justin/code/AUVSL_ROS/src/auvsl_control/src/anfis_ddpg.model')
    
    rospy.init_node('check_odometry')
    control_server = rospy.Service("anfis_control", AnfisControl, handleControl)
    rospy.spin()
        
    
    #torch.save(agent,'anfis_ddpg_trained.model')
    
