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
def test_course():
    path = [[0.0, 0.0],[2.0, 0.0], [2.0, 2.0], [4.0, 2.0]]
    print(path[0])
    print(path[0][0])
    agent = torch.load('ddpg.model')
    return path[0]
test_course()
