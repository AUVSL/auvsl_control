import rospy
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import time
import matplotlib.pyplot as plt
from numpy.linalg import inv
import math
import torch
from torch.utils.data import TensorDataset, DataLoader

import anfis
from membership import TrapezoidalMembFunc, make_trap_mfs, make_bell_mfs, BellMembFunc, Zero, make_zero
from experimental import train_anfis, test_anfis
dtype = torch.float


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

robot_path = []
dis_error = []


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

def test_course2():
    path = [
    [     0 ,        0],
    [14.7800,         0],
    [15.6427,   -0.3573],
    [16.0000,   -1.2200],
    [16.0000,   -4.7800],
    [15.6427,   -5.6427],
    [14.7800,   -6.0000],
    [1.2200 ,  -6.0000],
    [0.3573 ,  -6.3573],
    [0.0000 ,  -7.2200],
    [0.0000 , -10.7800],
    [0.3573 , -11.6427],
    [1.2200 , -12.0000],
    [14.7800,  -12.0000],
    [15.6427,  -12.3573],
    [16.0000,  -13.2200],
    [16.0000,  -16.7800],
    [15.6427,  -17.6427],
    [14.7800,  -18.0000],
    [-3.7800,  -18.0000],
    [-4.6427,  -17.6427],
    [-5.0000,  -16.7800],
    [-5.0000,   -1.2200],
    [-4.6427,   -0.3573],
    [-3.7800,         0],
    [     0 ,        0]

    ]
    return path


def test_course():
    path = [
        [0.0, 0.0],
        [1.0000  ,       0],
        [1.7000,   -0.1876],
        [2.2124,   -0.7000],
        [2.4000,   -1.4000],
        [2.5876,   -2.1000],
        [3.1000,   -2.6124],
        [3.8000,   -2.8000],
        [4.5000,   -2.6124],
        [5.0124,   -2.1000],
        [5.2000,   -1.4000],
        [5.3876,   -0.7000],
        [5.9000,   -0.1876],
        [6.6000,         0],
        [7.3000,   -0.1876],
        [7.8124,   -0.7000],
        [8.0000,   -1.4000],
        [8.1876,   -2.1000],
        [8.7000,   -2.6124],
        [9.4000,   -2.8000],
       [10.1000,   -2.6124],
       [10.6124,   -2.1000],
       [10.8000,   -1.4000],
       [10.9876,   -0.7000],
       [11.5000,   -0.1876],
       [12.2000,         0],
       [12.9000,   -0.1876],
       [13.4124,   -0.7000],
       [13.6000,   -1.4000],
       [13.7876,   -2.1000],
       [14.3000,   -2.6124],
       [15.0000,   -2.8000],
       [16.0000,   -2.8000] ]
#    path = [[0.0, 0.0],[2.0, 0.0], [2.0, 2.0], [4.0, 2.0]]
    return path

def test_course1():
    path = [
    [     0 ,        0],
    [5,         0],
    [5,   -5],
    [10,   -5],
    [10,   5],
    [15,   5],
    [15,   0],
    [20,   0]


    ]
    return path
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

def callback(msg):
    global x
    global y
    global q1
    global q2
    global q3
    global q4
    global currentAngle
    global robot_path
    global stop
    x = msg.pose.pose.position.x
    y = msg.pose.pose.position.y
    q2 = msg.pose.pose.orientation.x
    q3 = msg.pose.pose.orientation.y
    q4 = msg.pose.pose.orientation.z
    q1 = msg.pose.pose.orientation.w
    currentAngle = math.atan2(2*(q1*q4+q2*q3),1-2*(q3**2+q4**2))

    if stop == False:
        robot_path.append([x,y])
    print('x position: ',x)
    print('y position: ',y)

test_path = test_course()      ####testcoruse MUST start with 0,0 . Check this out
pathcount = 0
pathlength = len(test_path)
test_path.append([1000,1000])


###For ddpg , python 3
#agent = torch.load('anfis_ddpg.model')
#new_state = np.array([0.0,0.0,0.0])
#control_law = np.array([(agent.get_action(new_state))])
#print(control_law)

model= torch.load('models/anfis_model1.npy')

##########################################################3
rospy.init_node('check_odometry')
sub = rospy.Subscriber("/odom", Odometry, callback)

pub = rospy.Publisher("/cmd_vel",Twist,queue_size =10)
#    print(path_count)#

rate = rospy.Rate(100)

######################################################3

while not rospy.is_shutdown():
    current_point, target_point, future_point = target_generator(test_path)
    if stop == True:
        print("STOP")
        break
    new_state = fuzzy_error(current_point, target_point, future_point)
#   for ddpg model
#    control_law = np.array([(agent.get_action(new_state))])
#    control_law = control_law.item()

## For normal anfis model
#    control_law = torch.tensor([[0.0,0.0,0.0]])
    control_law = torch.tensor([new_state])
    control_law = model(control_law).item() * 6.
    if (control_law > 4):
        control_law = 4
    if (control_law < -4):
        control_law = -4

    twist_msg = Twist()
    twist_msg.linear.x = 2 
    twist_msg.angular.z = control_law 

    #plt.scatter(x, y)
    #plt.pause(0.00000000000000000000000000000000000001)



    pub.publish(twist_msg)
    rate.sleep()

#plt.show()
####plot
test_path = np.array(test_path)
robot_path = np.array(robot_path)
plt.plot(test_path[:-1,0], test_path[:-1,1])
plt.plot(robot_path[:,0], robot_path[:,1])
plt.show()

###distance error mean
#print(np.mean(dis_error))
