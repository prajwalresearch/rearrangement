import pickle
import numpy
import time

from networks.delan import DeepLagrangianNetwork
from networks.feedforward import FNN
import numpy as np
from scipy.io import loadmat
import argparse
import matplotlib.pyplot as plt
#from tqdm import tqdm # Displays a progress bar
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
#import wandb
import MetricBased
import simple_euclid_IK
#from main_tsp import get_joint_trajc

def infer_delan(start, goal, model, device, network='delan'):
    model.eval()
    MSEs = []
    i = 0
    with torch.no_grad():

        if network == 'delan':
            start = np.array([start[0], start[1], start[2]], dtype=np.float64)
            goal = np.array([goal[0], goal[1], goal[2]], dtype = np.float64)
            net_input = np.concatenate((start, goal))
            net_input = torch.from_numpy(net_input)

            start = torch.from_numpy(np.array([start[0],start[1], start[2]], dtype=np.float64))
            goal = torch.from_numpy(np.array([goal[0], goal[1], goal[2]], dtype=np.float64))

            x = start
            y = goal
            state = net_input

            state = state.to(device)
            x = x.to(device)
            y = y.to(device)
            new_out_H = model(state.float())
            new_out_H = np.squeeze(new_out_H)
            x = torch.reshape(x, (-1, 3))
            y = torch.reshape(y, (-1, 3))
            diff_1 = x-y
            prod_1 = diff_1.view(x.shape[0], -1, 3)
            prod_2 = diff_1.view(x.shape[0], 3, -1)
            pred_cost =  torch.matmul(torch.matmul((prod_1).double(),new_out_H.double()), prod_2.double())

            pred_cost = torch.squeeze(pred_cost)
            # print("harish_1")
            # print(pred_cost)
            return pred_cost.item()
        else :
            start = np.array([start[0], start[1], start[2]], dtype=np.float64)
            goal = np.array([goal[0], goal[1], goal[2]], dtype = np.float64)
            net_input = np.concatenate((start, goal))
            net_input = torch.from_numpy(net_input)

            start = torch.from_numpy(np.array([start[0],start[1], start[2]], dtype=np.float64))
            goal = torch.from_numpy(np.array([goal[0], goal[1], goal[2]], dtype=np.float64))

            x = start
            y = goal
            state = net_input

            state = state.to(device)
            x = x.to(device)
            y = y.to(device)
            new_out_H = model(state.float())

            return new_out_H.item()




allowed_error = 0.01

def check(f1, f2):
    diff = f1-f2
    return abs(diff[0]) < allowed_error and abs(diff[1]) < allowed_error

class Object():
    def __init__(self, start, goal, cost):
        self.start = numpy.array(start, dtype=numpy.float32)[0:2]
        self.goal = numpy.array(goal, dtype=numpy.float32)[0:2]
        self.cost_e = numpy.linalg.norm(self.goal - self.start)
        self.cost_j = cost

    def match_get_cost(self, start, goal):

        device = 'cpu'
        q_dim = 4
        hidden_size = 64
        # model = DeepLagrangianNetwork(q_dim, hidden_size, device=device).to(device)
        # PATH = "./memory_shuffle/weights/model_2_model_epoch_8000.pth"
        model.load_state_dict(torch.load(PATH, map_location=torch.device('cpu')))
        infer = infer_delan
        return infer(start, goal, model, device, network='fnn')





class Dataset():
    def __init__(self):
        self.objects = []

    def add_object(self, object):
        self.objects.append(object)

    def get_cost(self, start, goal):
        c = None
        for obj in self.objects:
            cost = obj.match_get_cost(start, goal)
            if cost is not None:
                c = cost
                break
        return c

    def generate_dist_matrix(self, points):
        a = numpy.zeros(shape=(len(points),len(points)))
        i = 0
        j = 0
        for point in points:
            j = 0
            for p in points:
                cost = self.get_cost(point, p)
                a[i][j] = cost
                j += 1
            i += 1
        return a

def load_obj(name):
    with open(name, 'rb') as f:
        return pickle.load(f)

def pprint(A):
    if A.ndim==1:
        print(A)
    else:
        w = max([len(str(s)) for s in A])
        print(u'\u250c'+u'\u2500'*w+u'\u2510')
        for AA in A:
            print(' ', end='')
            print('[', end='')
            for i,AAA in enumerate(AA[:-1]):
                w1=max([len(str(s)) for s in A[:,i]])
                print(str(AAA)+' '*(w1-len(str(AAA))+1),end='')
            w1=max([len(str(s)) for s in A[:,-1]])
            print(str(AA[-1])+' '*(w1-len(str(AA[-1]))),end='')
            print(']')
        print(u'\u2514'+u'\u2500'*w+u'\u2518')


def match_get_cost(start, goal, network='delan'):

    device = 'cpu'
    q_dim = 6
    hidden_size = 64
    if network == 'delan':
        model = DeepLagrangianNetwork(q_dim, hidden_size, device=device).to(device)
        PATH = "models/model_750_model_epoch_20000.pth"
    else :
        model = FNN(6, 1)
        PATH = "models/model_1500_model_epoch_5000.pth"
    model.load_state_dict(torch.load(PATH, map_location=torch.device('cpu')))
    infer = infer_delan
    return infer(start, goal, model, device, network)

def generate_dist_matrix(points, network='delan'):
    a = numpy.zeros(shape=(len(points),len(points)))
    i = 0
    j = 0
    for point in points:
        j = 0
        for p in points:
            cost = match_get_cost(point, p, network)
            a[i][j] = cost
            j += 1
        i += 1
    return a

def get_joint_distance_matrix(nodes, network ='delan'):
    # data = load_obj('/home/pushkalkatara/Downloads/data.pkl')

    # dataset = Dataset()
    # for id, d in data.items():
    #     o = Object(d['start_position'], d['goal_position'], d['cost_j'])
    #     dataset.add_object(o)
    dm = generate_dist_matrix(nodes, network)
    #pprint(dm)
    return dm




def calculate_joint_cost(route, planner='simple'):
    # data = load_obj('/home/pushkalkatara/Downloads/data.pkl')

    # dataset = Dataset()
    # for id, d in data.items():
    #     o = Object(d['start_position'], d['goal_position'], d['cost_j'])
    #     dataset.add_object(o)

    total_cost_j = 0.0
    total_cost_e = 0.0
    print("route : ", route)

    if planner != 'robotsp' :
        for i in range(len(route) - 1):
            st= time.time()
            print(route[i])
            # (x1, y1, z1) --> (x2, y2, z2)
            if i==0:
                init_joint =np.asarray([0.84382422 ,-1.75234566 ,-1.69086123 ,-2.25043335, 0.57998683,  1.20356413, -2.5246048 ])

            print(init_joint.shape)
            if planner == 'simple':
                JointCost, CartCost, cost_tracker, trajectory = simple_euclid_IK.findTrajectoryIK(init_joint, np.array(route[i]), np.array(route[i+1]))
            elif planner=='metric':
                JointCost, CartCost, cost_tracker, trajectory = MetricBased.trajMetricBased(init_joint, np.array(route[i]), np.array(route[i+1]))

            init_joint = trajectory[-1]
            # cost = match_get_cost(route[i], route[i+1])
            print("Network time:", time.time()-st)
            # cost = dataset.get_cost(route[i], route[i+1])
            total_cost_j += JointCost
            total_cost_e += CartCost

    else :
        total_cost_j, total_cost_e, trajectory = get_joint_trajc(route)


    return total_cost_j, total_cost_e
