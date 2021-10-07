#import rospy
#from nav_msgs.msg import Odometry
#from geometry_msgs.msg import Twist
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import time
import matplotlib.pyplot as plt
from numpy.linalg import inv
from torch.utils.data import TensorDataset, DataLoader

import anfis
from membership import TrapezoidalMembFunc, make_trap_mfs, make_bell_mfs, BellMembFunc, Zero, make_zero
from experimental import train_anfis, test_anfis
import sys

from ddpg import DDPGagent

from model import *
dtype = torch.float

model= torch.load('anfis_initial.model1')
#model= torch.load('p2_p3_nodown.model')
print("Tewdes")

print(model.get_action(np.array([0.0,0.0,0.0])))
#new_state = np.array([0.0,0.0,0.0])
#action = model.get_action(new_state)
#print(action)
