
import jax
import jax.numpy as jnp
import numpy as np
from jax import jit, vmap, random
import time

import roboticstoolbox as rtb
import cvxopt

robot = rtb.models.DH.Panda()
import autograd.numpy as np
import pdb
import cvxopt
from cvxopt import solvers
# import random
import matplotlib.pyplot as plt
from spatialmath import SE3
import torch
from dataset_custom_z import TrajectoryDataset_customScipy
from torch.utils.data import DataLoader
from autograd import grad

# # loading Valuefunction

# In[2]:
goal = None

device = 'cpu'
model = torch.load('model_750_model_epoch_20000.pth', map_location=torch.device('cpu'))  # loaded trained model
q_dim = 6  # q_dim is the dimension of joint space
q_dim_changed = int(0.5 * q_dim)
weight = []
for key in (model.keys()):
    # print(key)
    weight.append(model[key].cpu().numpy())  # load weight and bias


def leaky_relu(z):
    return jnp.maximum(0.01 * z, z)


def softplus(z, beta=1):
    return (1 / beta) * jnp.log(1 + jnp.exp(z * beta))


def assemble_lower_triangular_matrix(Lo, Ld):
    Lo = Lo.squeeze(0)
    Ld = Ld.squeeze(0)

    assert (2 * Lo.shape[0] == (Ld.shape[0] ** 2 - Ld.shape[0]))
    # pdb.set_trace()
    #     diagonal_matrix = np.diagflat(Ld)
    diagonal_matrix = jnp.identity(len(Ld)) * jnp.outer(jnp.ones(len(Ld)), Ld)

    L = jnp.tril(jnp.ones(diagonal_matrix.shape)) - jnp.eye(q_dim_changed)

    # Set off diagonals

    L = jnp.array([[0, 0, 0], [1, 0, 0], [0, 0, 0]]) * Lo.reshape(3)[0] + jnp.array([[0, 0, 0], [0, 0, 0], [1, 0, 0]]) * \
        Lo.reshape(3)[1] + jnp.array([[0, 0, 0], [0, 0, 0], [0, 1, 0]]) * Lo.reshape(3)[2]
    # print("L now : ", L)
    # Add diagonals
    L = L + diagonal_matrix
    return L


def value_func(x1):
    global weight, goal
    fc1_w = weight[0]
    fc1_b = weight[1]
    fc2_w = weight[2]
    fc2_b = weight[3]
    fc_Ld_w = weight[4]
    fc_Ld_b = weight[5]
    fc_Lo_w = weight[6]
    fc_Lo_b = weight[7]
    
    net_input = jnp.concatenate([x1, jnp.squeeze(goal)], axis =0)
    net_input = jnp.array([net_input])

    z1 = jnp.dot(net_input, fc1_w.transpose()) + fc1_b
    hidden1 = leaky_relu(z1)
    z2 = jnp.dot(hidden1, fc2_w.transpose()) + fc2_b
    hidden2 = leaky_relu(z2)
    hidden3 = jnp.dot(hidden2, fc_Ld_w.transpose()) + fc_Ld_b
    Ld = softplus(hidden3)
    Lo = jnp.dot(hidden2, fc_Lo_w.transpose()) + fc_Lo_b
    L = assemble_lower_triangular_matrix(Lo, Ld)

    H = L @ L.transpose() + 1e-9 * jnp.eye(3)

    # calculating value funcion ie optimal cost   (x1-x2).T*H*(x1-x2)
    x1 = jnp.reshape(net_input[0][:3], (-1, 3))
    goal = jnp.reshape(net_input[0][3:], (-1, 3))
    diff = x1 - goal
    optimalCost = jnp.matmul(diff, jnp.matmul(H, diff.transpose()))
    return optimalCost[0][0]

value_func = jit(vmap(value_func, in_axes=(1)))

def generate_gaussian(mean, cov):
 
  qdot_1,qdot_2, qdot_3, qdot_4, qdot_5, qdot_6, qdot_7 = jax.random.multivariate_normal(key, mean, cov, (num_samples,)).T
  return jnp.vstack(( qdot_1, qdot_2, qdot_3, qdot_4, qdot_5, qdot_6, qdot_7  ))



def fk_franka(q_1, q_2, q_3, q_4, q_5, q_6, q_7):
   
    x = -0.107*(((jnp.sin(q_1)*jnp.sin(q_3) - jnp.cos(q_1)*jnp.cos(q_2)*jnp.cos(q_3))*jnp.cos(q_4) - jnp.sin(q_2)*jnp.sin(q_4)*jnp.cos(q_1))*jnp.cos(q_5) + (jnp.sin(q_1)*jnp.cos(q_3) + jnp.sin(q_3)*jnp.cos(q_1)*jnp.cos(q_2))*jnp.sin(q_5))*jnp.sin(q_6) - 0.088*(((jnp.sin(q_1)*jnp.sin(q_3) - jnp.cos(q_1)*jnp.cos(q_2)*jnp.cos(q_3))*jnp.cos(q_4) - jnp.sin(q_2)*jnp.sin(q_4)*jnp.cos(q_1))*jnp.cos(q_5) + (jnp.sin(q_1)*jnp.cos(q_3) + jnp.sin(q_3)*jnp.cos(q_1)*jnp.cos(q_2))*jnp.sin(q_5))*jnp.cos(q_6) + 0.088*((jnp.sin(q_1)*jnp.sin(q_3) - jnp.cos(q_1)*jnp.cos(q_2)*jnp.cos(q_3))*jnp.sin(q_4) + jnp.sin(q_2)*jnp.cos(q_1)*jnp.cos(q_4))*jnp.sin(q_6) - 0.107*((jnp.sin(q_1)*jnp.sin(q_3) - jnp.cos(q_1)*jnp.cos(q_2)*jnp.cos(q_3))*jnp.sin(q_4) + jnp.sin(q_2)*jnp.cos(q_1)*jnp.cos(q_4))*jnp.cos(q_6) + 0.384*(jnp.sin(q_1)*jnp.sin(q_3) - jnp.cos(q_1)*jnp.cos(q_2)*jnp.cos(q_3))*jnp.sin(q_4) + 0.0825*(jnp.sin(q_1)*jnp.sin(q_3) - jnp.cos(q_1)*jnp.cos(q_2)*jnp.cos(q_3))*jnp.cos(q_4) - 0.0825*jnp.sin(q_1)*jnp.sin(q_3) - 0.0825*jnp.sin(q_2)*jnp.sin(q_4)*jnp.cos(q_1) + 0.384*jnp.sin(q_2)*jnp.cos(q_1)*jnp.cos(q_4) + 0.316*jnp.sin(q_2)*jnp.cos(q_1) + 0.0825*jnp.cos(q_1)*jnp.cos(q_2)*jnp.cos(q_3)

    y = 0.107*(((jnp.sin(q_1)*jnp.cos(q_2)*jnp.cos(q_3) + jnp.sin(q_3)*jnp.cos(q_1))*jnp.cos(q_4) + jnp.sin(q_1)*jnp.sin(q_2)*jnp.sin(q_4))*jnp.cos(q_5) - (jnp.sin(q_1)*jnp.sin(q_3)*jnp.cos(q_2) - jnp.cos(q_1)*jnp.cos(q_3))*jnp.sin(q_5))*jnp.sin(q_6) + 0.088*(((jnp.sin(q_1)*jnp.cos(q_2)*jnp.cos(q_3) + jnp.sin(q_3)*jnp.cos(q_1))*jnp.cos(q_4) + jnp.sin(q_1)*jnp.sin(q_2)*jnp.sin(q_4))*jnp.cos(q_5) - (jnp.sin(q_1)*jnp.sin(q_3)*jnp.cos(q_2) - jnp.cos(q_1)*jnp.cos(q_3))*jnp.sin(q_5))*jnp.cos(q_6) - 0.088*((jnp.sin(q_1)*jnp.cos(q_2)*jnp.cos(q_3) + jnp.sin(q_3)*jnp.cos(q_1))*jnp.sin(q_4) - jnp.sin(q_1)*jnp.sin(q_2)*jnp.cos(q_4))*jnp.sin(q_6) + 0.107*((jnp.sin(q_1)*jnp.cos(q_2)*jnp.cos(q_3) + jnp.sin(q_3)*jnp.cos(q_1))*jnp.sin(q_4) - jnp.sin(q_1)*jnp.sin(q_2)*jnp.cos(q_4))*jnp.cos(q_6) - 0.384*(jnp.sin(q_1)*jnp.cos(q_2)*jnp.cos(q_3) + jnp.sin(q_3)*jnp.cos(q_1))*jnp.sin(q_4) - 0.0825*(jnp.sin(q_1)*jnp.cos(q_2)*jnp.cos(q_3) + jnp.sin(q_3)*jnp.cos(q_1))*jnp.cos(q_4) - 0.0825*jnp.sin(q_1)*jnp.sin(q_2)*jnp.sin(q_4) + 0.384*jnp.sin(q_1)*jnp.sin(q_2)*jnp.cos(q_4) + 0.316*jnp.sin(q_1)*jnp.sin(q_2) + 0.0825*jnp.sin(q_1)*jnp.cos(q_2)*jnp.cos(q_3) + 0.0825*jnp.sin(q_3)*jnp.cos(q_1)

    z = -0.107*((jnp.sin(q_2)*jnp.cos(q_3)*jnp.cos(q_4) - jnp.sin(q_4)*jnp.cos(q_2))*jnp.cos(q_5) - jnp.sin(q_2)*jnp.sin(q_3)*jnp.sin(q_5))*jnp.sin(q_6) - 0.088*((jnp.sin(q_2)*jnp.cos(q_3)*jnp.cos(q_4) - jnp.sin(q_4)*jnp.cos(q_2))*jnp.cos(q_5) - jnp.sin(q_2)*jnp.sin(q_3)*jnp.sin(q_5))*jnp.cos(q_6) + 0.088*(jnp.sin(q_2)*jnp.sin(q_4)*jnp.cos(q_3) + jnp.cos(q_2)*jnp.cos(q_4))*jnp.sin(q_6) - 0.107*(jnp.sin(q_2)*jnp.sin(q_4)*jnp.cos(q_3) + jnp.cos(q_2)*jnp.cos(q_4))*jnp.cos(q_6) + 0.384*jnp.sin(q_2)*jnp.sin(q_4)*jnp.cos(q_3) + 0.0825*jnp.sin(q_2)*jnp.cos(q_3)*jnp.cos(q_4) - 0.0825*jnp.sin(q_2)*jnp.cos(q_3) - 0.0825*jnp.sin(q_4)*jnp.cos(q_2) + 0.384*jnp.cos(q_2)*jnp.cos(q_4) + 0.316*jnp.cos(q_2) + 0.33

    return x, y, z

top_k = 100
num_samples = 1000

generate_gaussian_jit = jit(generate_gaussian)
fk_franka_jit = jit(fk_franka)

qdot = jnp.zeros((7, 1))
TRAJ_train = TrajectoryDataset_customScipy()
trainloader = DataLoader(TRAJ_train, batch_size=1, drop_last=True, shuffle=True)
for x, y, net_input, cost, start_joint, end_joint in trainloader:
    traj = []
    x = x.to(device).numpy()
    y = y.to(device).numpy()
    net_input = net_input.to(device).numpy()
    cost = cost.to(device).numpy()
    start_joint = jnp.squeeze(start_joint.to(device).numpy())
    end_joint = jnp.squeeze(end_joint.to(device).numpy())
    
    
    init_joint = start_joint
    start_goalMatrx = robot.fkine(np.asarray(init_joint)).A
    start_cartesian = start_goalMatrx[:3,-1]
    end_goalMatrx = robot.fkine(np.asarray(end_joint)).A
    end_cartesian = end_goalMatrx[:3,-1]
    goal = jnp.squeeze(end_cartesian)

    
    q_min = jnp.array([-165, -100, -165, -165, -165, -1.0, -165  ])*np.pi/180
    q_max = jnp.array([ 165,   101,  165,  1.0,    165, 214, 165  ])*np.pi/180

    q_min_jax = jnp.vstack((q_min[0]*jnp.ones(num_samples), q_min[1]*jnp.ones(num_samples), q_min[2]*jnp.ones(num_samples), q_min[3]*jnp.ones(num_samples), q_min[4]*jnp.ones(num_samples), q_min[5]*jnp.ones(num_samples), q_min[6]*jnp.ones(num_samples)            ))

    q_max_jax = jnp.vstack((q_max[0]*jnp.ones(num_samples), q_max[1]*jnp.ones(num_samples), q_max[2]*jnp.ones(num_samples), q_max[3]*jnp.ones(num_samples), q_max[4]*jnp.ones(num_samples), q_max[5]*jnp.ones(num_samples), q_max[6]*jnp.ones(num_samples)            ))

    qdot_min = -1.5*jnp.ones(( 7, num_samples   ))
    qdot_max = 1.5*jnp.ones(( 7, num_samples   ))



    key = random.PRNGKey(0)
    t = 0.01
    q = start_joint

    
    for j in range(1000):
        cov = 5*jnp.identity(7)
        mean = 0*jnp.ones(7)

        for i in range(0, 5):

            qdot_samples = generate_gaussian_jit(mean, cov)

            qdot_samples = jnp.clip(qdot_samples, qdot_min, qdot_max) #######3 kinematically feasible velocities

            
            q_samples = qdot_samples*t+q[:, jnp.newaxis]
            q_samples = jnp.clip(q_samples, q_min_jax, q_max_jax  ) ################## clipping to min and max values


            x_samples, y_samples, z_samples = fk_franka_jit(q_samples[0,:], q_samples[1,:], q_samples[2,:], q_samples[3,:], q_samples[4,:], q_samples[5,:], q_samples[6,:])
            
            
            net_input = jnp.concatenate((x_samples.reshape(1, num_samples), y_samples.reshape(1, num_samples), z_samples.reshape(1, num_samples)), axis=0)
            value_samples = value_func(net_input)
            
            
            samples_losses = value_samples+0.0001*jnp.linalg.norm((qdot_samples-qdot).T, axis = 1)

            sorted_index = np.argsort(samples_losses)

            mean = jnp.mean(qdot_samples.T[sorted_index[0:top_k]], axis=0)
            # cov = jnp.diag(jnp.std(qdot_samples.T[sorted_index[0:top_k]], axis=0))
            cov = jnp.cov(qdot_samples.T[sorted_index[0:top_k]].T).T
            
            print("LOSS : ", samples_losses[sorted_index[0]])
            
            
            

        qdot = qdot_samples.T[sorted_index[0]]
        q = q+qdot*t
        
        x, y, z = fk_franka_jit(q[0], q[1], q[2], q[3], q[4], q[5], q[6])
        qdot = qdot.reshape(7, 1)
        print("ITERS : ", j )
        print(x, y, z)
        traj.append([x, y, z])
        
        print(end_cartesian)
        print(((x - end_cartesian[0])**2 + (y - end_cartesian[1])**2 +(z - end_cartesian[2])**2)**0.5)
        if ((x - end_cartesian[0])**2 + (y - end_cartesian[1])**2 +(z - end_cartesian[2])**2)**0.5 < 0.01 or j==999:
            ax = plt.axes(projection='3d')
            traj = np.array(traj)
            ax.view_init(30, 0)
            ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], label = "Generated Trajectory")
            ax.scatter(traj[:, 0], traj[:, 1], traj[:, 2])
            ax.scatter(end_cartesian[0], end_cartesian[1], end_cartesian[2], label = "goal", s=100)
            plt.legend()
            plt.show()
            print("Converged")
            break
        print()



    print(jnp.shape(x))






# print(np.shape(qdot))
