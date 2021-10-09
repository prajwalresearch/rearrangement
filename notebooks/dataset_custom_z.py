import numpy as np
import torch
import pandas as pd
from torch.utils.data.dataset import Dataset
import pickle
import pdb

#  goal - (x1, y1)
#  start - (x2, y2)
#  cost_joint - 2.34
class TrajectoryDataset_customScipy(Dataset):

    def __init__(self):
        self.training_data= []
        with open("dataScipy.pkl", 'rb') as f:
            try:
                while True:
                    data = pickle.load(f)
                    self.training_data.append(data)
            except:
                pass
        f.close()
        with open("dataScipy2.pkl", 'rb') as f:
            try:
                while True:
                    data = pickle.load(f)
                    self.training_data.append(data)
            except:
                pass
        f.close()
        self.itr =0

    def __len__(self):
        return len(self.training_data)

    def __getitem__(self, index):
        #pdb.set_trace()
        start = np.array(
            [self.training_data[index]['start_position'][0], self.training_data[index]['start_position'][1],
             self.training_data[index]['start_position'][2]], dtype=np.float64)
        goal = np.array(
            [self.training_data[index]['goal_position'][0], self.training_data[index]['goal_position'][1],
             self.training_data[index]['goal_position'][2]], dtype=np.float64)

        net_input = np.concatenate((start, goal))
        net_input = torch.from_numpy(net_input)

        start = torch.from_numpy(start)
        goal = torch.from_numpy(goal)

        cost = self.training_data[index]['cost_j']

        start_joint = torch.from_numpy(np.array(list(self.training_data[index]['trajectory'][:,0])))
        end_joint = torch.from_numpy(np.array(list(self.training_data[index]['trajectory'][:,-1])))
        #pdb.set_trace()
        return start, goal, net_input, cost, start_joint, end_joint

class TrajectoryDataset_custom(Dataset):

    def __init__(self, file_path):
        with open(file_path, 'rb') as f:
            self.training_data = pickle.load(f)

    def __len__(self):
        return len(self.training_data)

    def __getitem__(self, index):
        #pdb.set_trace()
        start = np.array(
            [self.training_data[index]['start_position'][0], self.training_data[index]['start_position'][1],
             self.training_data[index]['start_position'][2]], dtype=np.float64)
        goal = np.array(
            [self.training_data[index]['goal_postiion'][0], self.training_data[index]['goal_postiion'][1],
             self.training_data[index]['goal_postiion'][2]], dtype=np.float64)

        net_input = np.concatenate((start, goal))
        net_input = torch.from_numpy(net_input)

        # start = torch.from_numpy(np.array([self.training_data[index][0][0], self.training_data[index][0][1]], dtype=np.float64))
        # goal = torch.from_numpy(np.array([self.training_data[index][1][0], self.training_data[index][1][1]], dtype=np.float64))

        cost = self.training_data[index]['cost_j']

        start_joint = torch.from_numpy(np.array(list(self.training_data[index]['trajectory'][:,0])))
        end_joint = torch.from_numpy(np.array(list(self.training_data[index]['trajectory'][:,:-1])))

        return start, goal, net_input, cost, start_joint, end_joint


class TrajectoryDataset_custom_old_1(Dataset):

    def __init__(self, file_path):
        # self.training_file = pd.read_csv(file_path)
        # self.start_x = self.training_file['start_x']
        # self.start_y = self.training_file['start_y']
        # self.start_theta1 = self.training_file['start_theta1']
        # self.start_theta2 = self.training_file['start_theta2']
        # self.goal_x = self.training_file['goal_x']
        # self.goal_y = self.training_file['goal_y']
        # self.goal_theta1 = self.training_file['goal_theta1']
        # self.goal_theta2= self.training_file['goal_theta2']

        with open(file_path, 'rb') as f:
            self.training_data = pickle.load(f)

    # def load_obj(name):

    def __len__(self):
        return len(self.training_data)

    def __getitem__(self, index):
        start = np.array([self.training_data[index][0][0], self.training_data[index][0][1]], dtype=np.float64)
        goal = np.array([self.training_data[index][1][0], self.training_data[index][1][1]], dtype=np.float64)

        net_input = np.concatenate((start, goal))
        net_input = torch.from_numpy(net_input)

        start = torch.from_numpy(
            np.array([self.training_data[index][0][0], self.training_data[index][0][1]], dtype=np.float64))
        goal = torch.from_numpy(
            np.array([self.training_data[index][1][0], self.training_data[index][1][1]], dtype=np.float64))

        cost = self.training_data[index][2]

        # if cost is None:
        #     cost = np.array([1000000000], dtype = np.float64)
        #     pass

        return start, goal, net_input, cost


class TrajectoryDataset_custom_old(Dataset):

    def __init__(self, file_path):
        self.training_file = pd.read_csv(file_path)
        self.start_x = self.training_file['start_x']
        self.start_y = self.training_file['start_y']
        self.start_theta1 = self.training_file['start_theta1']
        self.start_theta2 = self.training_file['start_theta2']
        self.goal_x = self.training_file['goal_x']
        self.goal_y = self.training_file['goal_y']
        self.goal_theta1 = self.training_file['goal_theta1']
        self.goal_theta2 = self.training_file['goal_theta2']

    def __len__(self):
        return len(self.start_x)

    def __getitem__(self, index):
        start = np.array([self.start_x[index], self.start_y[index]], dtype=np.float64)
        goal = np.array([self.goal_x[index], self.goal_y[index]], dtype=np.float64)

        net_input = np.concatenate((start, goal))
        net_input = torch.from_numpy(net_input)

        start = torch.from_numpy(np.array([self.start_x[index], self.start_y[index]], dtype=np.float64))
        goal = torch.from_numpy(np.array([self.goal_x[index], self.goal_y[index]], dtype=np.float64))

        start_theta = np.array([self.start_theta1[index], self.start_theta2[index]], dtype=np.float64)
        goal_theta = np.array([self.goal_theta1[index], self.goal_theta2[index]], dtype=np.float64)
        cost = np.linalg.norm(start_theta - goal_theta)

        return start, goal, net_input, cost