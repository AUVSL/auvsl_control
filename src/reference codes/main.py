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
def vels(angular_vel,width,speed):
    vl = speed - 0.5*width*angular_vel
    vr = speed + 0.5*width*angular_vel
    return vl,vr
def phy(vl,vr,width,prev_vel,prev_theta,prev_x,prev_y,prev_w):

    theta = prev_theta + (0.001* ( ((vr - vl) / width) - prev_w )/2) ###do integral
    print(theta)
    x = prev_x + ( ((vr+vl)/2 - prev_vel)/2 * np.cos(theta)) *0.001
    y = prev_y + ( ((vr+vl)/2 - prev_vel)/2 * np.sin(theta)) *0.001
    prev_vel = (vr+vl)/2
    prev_theta = theta
    prev_x = x
    prev_y = y
    prev_w = ((vr - vl) / width)
    return [theta, x, y, prev_vel,prev_theta,prev_x,prev_y,prev_w]
if __name__ == '__main__':
    model = torch.load('anfis_model.npy')
    path_x = [0,5,5,10,10,15,15,20]
    path_y = [0,0,-5,-5,5,5,0,0]
    pathcount = 0
    pathlength = len(path_x)
    path_x.append(200)
    path_y.append(200)
    speed = 2
    width = 0.323
    u = [0,0,0]
    stop = False
    prev_vel = 0
    prev_theta = 0
    prev_x = 0
    prev_y = 0
    prev_w = 0
    robot_path_x = [0]
    robot_path_y = [0]
    i = 0
    while(stop == False):
        pos = [u[1],u[2]]
        print(pos)
        current_angle = wraptopi(u[0])
        current_point = np.array([path_x[pathcount],path_y[pathcount]])
        target = np.array([path_x[pathcount+1],path_y[pathcount+1]])
    #    distErr = np.sqrt((target[0]-pos[0])**2+(target[1]-pos[1])**2)
        A = np.array([ [(current_point[1]-target[1]),(target[0]-current_point[0])], [(target[0]-current_point[0]), (target[1]-current_point[1])] ])
        b = np.array([ [(target[0]*current_point[1] - current_point[0]*target[1])], [(pos[0]*(target[0]-current_point[0]) + pos[1]*(target[1] - current_point[1]))] ])
        proj = inv(A)*b
        projLen = np.dot(proj-current_point,target-current_point).sum()  / np.linalg.norm(target - current_point,2)**2
        if (projLen > 1):
            pathcount += 1
            print(pathcount)
            current_point = np.array([path_x[pathcount],path_y[pathcount]])
            target = np.array([path_x[pathcount+1],path_y[pathcount+1]])
        if (pathcount == pathlength-1):
            stop = True
            break

        if ( (pathcount == (pathlength-2)) or (pathcount == (pathlength -1)) ):
            a = np.array([path_x[pathcount],path_y[pathcount]])
            b = np.array([path_x[pathcount+1],path_y[pathcount+1]])
            post = np.array([path_x[pathcount+1],path_y[pathcount+1]])
        else:
            a = np.array([path_x[pathcount],path_y[pathcount]])
            b = np.array([path_x[pathcount+1],path_y[pathcount+1]])
            post = np.array([path_x[pathcount+2],path_y[pathcount+1]])
        th1 = math.atan2(b[1]-pos[1], b[0]-pos[0])
        th2 = math.atan2(b[1]-a[1], b[0]-a[0])
        th3 = math.atan2(post[1]-b[1], post[0]-b[0])
        theta_far = angdiff(current_angle,th1)
        theta_near = angdiff(current_angle,th2)
        d = (pos[0] - current_point[0]) * (target[1]-current_point[1]) - (pos[1]-current_point[1]) * (target[0] - current_point[0])
        if (d>0):
            side = 1
        elif (d<0):
            side = -1
        else:
            side = 0
        distanceError = np.linalg.norm(pos-proj,2) * side
    #    print([distanceError,theta_far,theta_near])
        x = torch.tensor([[distanceError, theta_far, theta_near]],dtype = torch.float)
        angular_vel = 5*model(x).item()
    #    print(angular_vel)
        if (angular_vel > np.pi):
            angular_vel = np.pi
        if (angular_vel < -np.pi):
            angular_vel = -np.pi
        vl, vr = vels(angular_vel,width,speed)
        u = phy(vl,vr,width,prev_vel,prev_theta,prev_x,prev_y,prev_w)
#        print(u)
        prev_vel = u[3]
        prev_theta = u[4]
        prev_x = u[5]
        prev_y = u[6]
        prev_w = u[7]
        #print([u[1],u[2]])
        robot_path_x.append(u[1])
        robot_path_y.append(u[2])
    #    i += 1
    #    if (i==1000):
    #        break
        #stop =True

    plt.plot(robot_path_x,robot_path_y)
    plt.show()

    x = torch.tensor([[0, 0, 0]])
    angular_vel = model(x)
    print(angular_vel)
    print(angdiff(0.4314,1.643))
