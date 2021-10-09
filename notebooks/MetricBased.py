import cvxopt
import cvxopt
from cvxopt import solvers
import random
import matplotlib.pyplot as plt
import torch
from autograd import grad
from autograd import jacobian
import autograd.numpy as np
import autograd.numpy as jnp
import scipy.optimize as optim
from scipy.optimize import minimize, Bounds,LinearConstraint
from scipy.optimize import LinearConstraint,NonlinearConstraint
from scipy.optimize import BFGS


t = 0.02
q_prev = None

device = 'cpu'
model = torch.load('models/model_750_model_epoch_20000.pth', map_location=torch.device('cpu'))  # loaded trained model
q_dim = 6  # q_dim is the dimension of joint space
q_dim_changed = int(0.5 * q_dim)


#value function defnation
weight = []
for key in (model.keys()):
    # print(key)
    weight.append(model[key].cpu().numpy())  # load weight and bias


def leaky_relu(z):
    return np.maximum(0.01 * z, z)


def softplus(z, beta=1):
    return (1 / beta) * np.log(1 + np.exp(z * beta))


def assemble_lower_triangular_matrix(Lo, Ld):
    Lo = Lo.squeeze(0)
    Ld = Ld.squeeze(0)

    assert (2 * Lo.shape[0] == (Ld.shape[0] ** 2 - Ld.shape[0]))
    # pdb.set_trace()
    #     diagonal_matrix = np.diagflat(Ld)
    diagonal_matrix = np.identity(len(Ld)) * np.outer(np.ones(len(Ld)), Ld)

    L = np.tril(np.ones(diagonal_matrix.shape)) - np.eye(q_dim_changed)

    # Set off diagonals

    L = np.array([[0, 0, 0], [1, 0, 0], [0, 0, 0]]) * Lo.reshape(3)[0] + np.array([[0, 0, 0], [0, 0, 0], [1, 0, 0]]) * \
        Lo.reshape(3)[1] + np.array([[0, 0, 0], [0, 0, 0], [0, 1, 0]]) * Lo.reshape(3)[2]
    # print("L now : ", L)
    # Add diagonals
    L = L + diagonal_matrix
    return L


def value(x1):
    global weight, goal
    fc1_w = weight[0]
    fc1_b = weight[1]
    fc2_w = weight[2]
    fc2_b = weight[3]
    fc_Ld_w = weight[4]
    fc_Ld_b = weight[5]
    fc_Lo_w = weight[6]
    fc_Lo_b = weight[7]
    #pdb.set_trace()
    net_input = np.concatenate([np.squeeze(x1), np.squeeze(goal)], axis=0)
    net_input = np.array([net_input])

    z1 = np.dot(net_input, fc1_w.transpose()) + fc1_b
    hidden1 = leaky_relu(z1)
    z2 = np.dot(hidden1, fc2_w.transpose()) + fc2_b
    hidden2 = leaky_relu(z2)
    hidden3 = np.dot(hidden2, fc_Ld_w.transpose()) + fc_Ld_b
    Ld = softplus(hidden3)
    Lo = np.dot(hidden2, fc_Lo_w.transpose()) + fc_Lo_b
    L = assemble_lower_triangular_matrix(Lo, Ld)

    H = L @ L.transpose() + 1e-9 * np.eye(3)
    return H

grad_value = grad(value)
jac_value= jacobian(value)

def fk_franka(q):
    q_1 = q[0]
    q_2 = q[1]
    q_3 = q[2]
    q_4 = q[3]
    q_5 = q[4]
    q_6 = q[5]
    q_7 = q[6]

    x = 0.0825 * jnp.cos(q_1) * jnp.cos(q_2) * jnp.cos(q_3) + 0.384 * jnp.cos(q_1) * jnp.cos(q_4) * jnp.sin(
        q_2) - 0.0825 * jnp.cos(q_1) * jnp.sin(q_2) * jnp.sin(q_4) + 0.316 * jnp.cos(q_1) * jnp.sin(
        q_2) - 0.0825 * jnp.cos(q_4) * (
                jnp.cos(q_1) * jnp.cos(q_2) * jnp.cos(q_3) - jnp.sin(q_1) * jnp.sin(q_3)) + 0.088 * jnp.cos(q_6) * (
                jnp.cos(q_5) * (jnp.cos(q_1) * jnp.sin(q_2) * jnp.sin(q_4) + jnp.cos(q_4) * (
                jnp.cos(q_1) * jnp.cos(q_2) * jnp.cos(q_3) - jnp.sin(q_1) * jnp.sin(q_3))) - jnp.sin(q_5) * (
                        jnp.cos(q_1) * jnp.cos(q_2) * jnp.sin(q_3) + jnp.cos(q_3) * jnp.sin(q_1))) - 0.21 * jnp.cos(
        q_6) * (
                jnp.cos(q_1) * jnp.cos(q_4) * jnp.sin(q_2) - jnp.sin(q_4) * (
                jnp.cos(q_1) * jnp.cos(q_2) * jnp.cos(q_3) - jnp.sin(q_1) * jnp.sin(q_3))) - 0.0825 * jnp.sin(
        q_1) * jnp.sin(q_3) - 0.384 * jnp.sin(q_4) * (
                jnp.cos(q_1) * jnp.cos(q_2) * jnp.cos(q_3) - jnp.sin(q_1) * jnp.sin(q_3)) + 0.21 * jnp.sin(q_6) * (
                jnp.cos(q_5) * (jnp.cos(q_1) * jnp.sin(q_2) * jnp.sin(q_4) + jnp.cos(q_4) * (
                jnp.cos(q_1) * jnp.cos(q_2) * jnp.cos(q_3) - jnp.sin(q_1) * jnp.sin(q_3))) - jnp.sin(q_5) * (
                        jnp.cos(q_1) * jnp.cos(q_2) * jnp.sin(q_3) + jnp.cos(q_3) * jnp.sin(q_1))) + 0.088 * jnp.sin(
        q_6) * (
                jnp.cos(q_1) * jnp.cos(q_4) * jnp.sin(q_2) - jnp.sin(q_4) * (
                jnp.cos(q_1) * jnp.cos(q_2) * jnp.cos(q_3) - jnp.sin(q_1) * jnp.sin(q_3)))
    y = 0.0825 * jnp.cos(q_1) * jnp.sin(q_3) + 0.0825 * jnp.cos(q_2) * jnp.cos(q_3) * jnp.sin(q_1) + 0.384 * jnp.cos(
        q_4) * jnp.sin(q_1) * jnp.sin(q_2) - 0.0825 * jnp.cos(q_4) * (
                jnp.cos(q_1) * jnp.sin(q_3) + jnp.cos(q_2) * jnp.cos(q_3) * jnp.sin(q_1)) + 0.088 * jnp.cos(q_6) * (
                jnp.cos(q_5) * (
                jnp.cos(q_4) * (jnp.cos(q_1) * jnp.sin(q_3) + jnp.cos(q_2) * jnp.cos(q_3) * jnp.sin(q_1)) + jnp.sin(
            q_1) * jnp.sin(q_2) * jnp.sin(q_4)) + jnp.sin(q_5) * (
                        jnp.cos(q_1) * jnp.cos(q_3) - jnp.cos(q_2) * jnp.sin(q_1) * jnp.sin(q_3))) - 0.21 * jnp.cos(
        q_6) * (jnp.cos(q_4) * jnp.sin(q_1) * jnp.sin(q_2) - jnp.sin(q_4) * (
            jnp.cos(q_1) * jnp.sin(q_3) + jnp.cos(q_2) * jnp.cos(q_3) * jnp.sin(q_1))) - 0.0825 * jnp.sin(
        q_1) * jnp.sin(q_2) * jnp.sin(q_4) + 0.316 * jnp.sin(q_1) * jnp.sin(q_2) - 0.384 * jnp.sin(q_4) * (
                jnp.cos(q_1) * jnp.sin(q_3) + jnp.cos(q_2) * jnp.cos(q_3) * jnp.sin(q_1)) + 0.21 * jnp.sin(q_6) * (
                jnp.cos(q_5) * (
                jnp.cos(q_4) * (jnp.cos(q_1) * jnp.sin(q_3) + jnp.cos(q_2) * jnp.cos(q_3) * jnp.sin(q_1)) + jnp.sin(
            q_1) * jnp.sin(q_2) * jnp.sin(q_4)) + jnp.sin(q_5) * (
                        jnp.cos(q_1) * jnp.cos(q_3) - jnp.cos(q_2) * jnp.sin(q_1) * jnp.sin(q_3))) + 0.088 * jnp.sin(
        q_6) * (
                jnp.cos(q_4) * jnp.sin(q_1) * jnp.sin(q_2) - jnp.sin(q_4) * (
                jnp.cos(q_1) * jnp.sin(q_3) + jnp.cos(q_2) * jnp.cos(q_3) * jnp.sin(q_1)))
    z = 0.384 * jnp.cos(q_2) * jnp.cos(q_4) - 0.0825 * jnp.cos(q_2) * jnp.sin(q_4) + 0.316 * jnp.cos(
        q_2) + 0.0825 * jnp.cos(q_3) * jnp.cos(q_4) * jnp.sin(q_2) + 0.384 * jnp.cos(q_3) * jnp.sin(q_2) * jnp.sin(
        q_4) - 0.0825 * jnp.cos(q_3) * jnp.sin(q_2) - 0.21 * jnp.cos(q_6) * (
                jnp.cos(q_2) * jnp.cos(q_4) + jnp.cos(q_3) * jnp.sin(q_2) * jnp.sin(q_4)) + 0.088 * jnp.cos(q_6) * (
                jnp.cos(q_5) * (jnp.cos(q_2) * jnp.sin(q_4) - jnp.cos(q_3) * jnp.cos(q_4) * jnp.sin(q_2)) + jnp.sin(
            q_2) * jnp.sin(q_3) * jnp.sin(q_5)) + 0.088 * jnp.sin(q_6) * (
                jnp.cos(q_2) * jnp.cos(q_4) + jnp.cos(q_3) * jnp.sin(q_2) * jnp.sin(q_4)) + 0.21 * jnp.sin(q_6) * (
                jnp.cos(q_5) * (jnp.cos(q_2) * jnp.sin(q_4) - jnp.cos(q_3) * jnp.cos(q_4) * jnp.sin(q_2)) + jnp.sin(
            q_2) * jnp.sin(q_3) * jnp.sin(q_5)) + 0.33
    cartpos = np.array([x,y,z])
    return cartpos
jac_fk = jacobian(fk_franka)


def traj_cost(trajectory):
    cost = 0
    cart_cost = 0
    for i in range(len(trajectory) - 1):
        cost += np.linalg.norm(np.asarray(trajectory[i]) - np.asarray(trajectory[i + 1]), ord=2)
        # pdb.set_trace()
        current = np.asarray(fk_franka(trajectory[i]))
        next = np.asarray(fk_franka(trajectory[i+1]))
        cart_cost += np.linalg.norm(current - next, ord=2)
    return cost, cart_cost


def constraintfxn(qdot_x_next ):
    global t, q_prev
    q = qdot_x_next[:7]*t  + q_prev
    q_1 = q[0]
    q_2 = q[1]
    q_3 = q[2]
    q_4 = q[3]
    q_5 = q[4]
    q_6 = q[5]
    q_7 = q[6]
    x_next = qdot_x_next[7]
    y_next = qdot_x_next[8]
    z_next = qdot_x_next[9]

    x = 0.0825 * jnp.cos(q_1) * jnp.cos(q_2) * jnp.cos(q_3) + 0.384 * jnp.cos(q_1) * jnp.cos(q_4) * jnp.sin(
        q_2) - 0.0825 * jnp.cos(q_1) * jnp.sin(q_2) * jnp.sin(q_4) + 0.316 * jnp.cos(q_1) * jnp.sin(
        q_2) - 0.0825 * jnp.cos(q_4) * (
                jnp.cos(q_1) * jnp.cos(q_2) * jnp.cos(q_3) - jnp.sin(q_1) * jnp.sin(q_3)) + 0.088 * jnp.cos(q_6) * (
                jnp.cos(q_5) * (jnp.cos(q_1) * jnp.sin(q_2) * jnp.sin(q_4) + jnp.cos(q_4) * (
                jnp.cos(q_1) * jnp.cos(q_2) * jnp.cos(q_3) - jnp.sin(q_1) * jnp.sin(q_3))) - jnp.sin(q_5) * (
                        jnp.cos(q_1) * jnp.cos(q_2) * jnp.sin(q_3) + jnp.cos(q_3) * jnp.sin(q_1))) - 0.21 * jnp.cos(
        q_6) * (
                jnp.cos(q_1) * jnp.cos(q_4) * jnp.sin(q_2) - jnp.sin(q_4) * (
                jnp.cos(q_1) * jnp.cos(q_2) * jnp.cos(q_3) - jnp.sin(q_1) * jnp.sin(q_3))) - 0.0825 * jnp.sin(
        q_1) * jnp.sin(q_3) - 0.384 * jnp.sin(q_4) * (
                jnp.cos(q_1) * jnp.cos(q_2) * jnp.cos(q_3) - jnp.sin(q_1) * jnp.sin(q_3)) + 0.21 * jnp.sin(q_6) * (
                jnp.cos(q_5) * (jnp.cos(q_1) * jnp.sin(q_2) * jnp.sin(q_4) + jnp.cos(q_4) * (
                jnp.cos(q_1) * jnp.cos(q_2) * jnp.cos(q_3) - jnp.sin(q_1) * jnp.sin(q_3))) - jnp.sin(q_5) * (
                        jnp.cos(q_1) * jnp.cos(q_2) * jnp.sin(q_3) + jnp.cos(q_3) * jnp.sin(q_1))) + 0.088 * jnp.sin(
        q_6) * (
                jnp.cos(q_1) * jnp.cos(q_4) * jnp.sin(q_2) - jnp.sin(q_4) * (
                jnp.cos(q_1) * jnp.cos(q_2) * jnp.cos(q_3) - jnp.sin(q_1) * jnp.sin(q_3)))
    y = 0.0825 * jnp.cos(q_1) * jnp.sin(q_3) + 0.0825 * jnp.cos(q_2) * jnp.cos(q_3) * jnp.sin(q_1) + 0.384 * jnp.cos(
        q_4) * jnp.sin(q_1) * jnp.sin(q_2) - 0.0825 * jnp.cos(q_4) * (
                jnp.cos(q_1) * jnp.sin(q_3) + jnp.cos(q_2) * jnp.cos(q_3) * jnp.sin(q_1)) + 0.088 * jnp.cos(q_6) * (
                jnp.cos(q_5) * (
                jnp.cos(q_4) * (jnp.cos(q_1) * jnp.sin(q_3) + jnp.cos(q_2) * jnp.cos(q_3) * jnp.sin(q_1)) + jnp.sin(
            q_1) * jnp.sin(q_2) * jnp.sin(q_4)) + jnp.sin(q_5) * (
                        jnp.cos(q_1) * jnp.cos(q_3) - jnp.cos(q_2) * jnp.sin(q_1) * jnp.sin(q_3))) - 0.21 * jnp.cos(
        q_6) * (jnp.cos(q_4) * jnp.sin(q_1) * jnp.sin(q_2) - jnp.sin(q_4) * (
            jnp.cos(q_1) * jnp.sin(q_3) + jnp.cos(q_2) * jnp.cos(q_3) * jnp.sin(q_1))) - 0.0825 * jnp.sin(
        q_1) * jnp.sin(q_2) * jnp.sin(q_4) + 0.316 * jnp.sin(q_1) * jnp.sin(q_2) - 0.384 * jnp.sin(q_4) * (
                jnp.cos(q_1) * jnp.sin(q_3) + jnp.cos(q_2) * jnp.cos(q_3) * jnp.sin(q_1)) + 0.21 * jnp.sin(q_6) * (
                jnp.cos(q_5) * (
                jnp.cos(q_4) * (jnp.cos(q_1) * jnp.sin(q_3) + jnp.cos(q_2) * jnp.cos(q_3) * jnp.sin(q_1)) + jnp.sin(
            q_1) * jnp.sin(q_2) * jnp.sin(q_4)) + jnp.sin(q_5) * (
                        jnp.cos(q_1) * jnp.cos(q_3) - jnp.cos(q_2) * jnp.sin(q_1) * jnp.sin(q_3))) + 0.088 * jnp.sin(
        q_6) * (
                jnp.cos(q_4) * jnp.sin(q_1) * jnp.sin(q_2) - jnp.sin(q_4) * (
                jnp.cos(q_1) * jnp.sin(q_3) + jnp.cos(q_2) * jnp.cos(q_3) * jnp.sin(q_1)))
    z = 0.384 * jnp.cos(q_2) * jnp.cos(q_4) - 0.0825 * jnp.cos(q_2) * jnp.sin(q_4) + 0.316 * jnp.cos(
        q_2) + 0.0825 * jnp.cos(q_3) * jnp.cos(q_4) * jnp.sin(q_2) + 0.384 * jnp.cos(q_3) * jnp.sin(q_2) * jnp.sin(
        q_4) - 0.0825 * jnp.cos(q_3) * jnp.sin(q_2) - 0.21 * jnp.cos(q_6) * (
                jnp.cos(q_2) * jnp.cos(q_4) + jnp.cos(q_3) * jnp.sin(q_2) * jnp.sin(q_4)) + 0.088 * jnp.cos(q_6) * (
                jnp.cos(q_5) * (jnp.cos(q_2) * jnp.sin(q_4) - jnp.cos(q_3) * jnp.cos(q_4) * jnp.sin(q_2)) + jnp.sin(
            q_2) * jnp.sin(q_3) * jnp.sin(q_5)) + 0.088 * jnp.sin(q_6) * (
                jnp.cos(q_2) * jnp.cos(q_4) + jnp.cos(q_3) * jnp.sin(q_2) * jnp.sin(q_4)) + 0.21 * jnp.sin(q_6) * (
                jnp.cos(q_5) * (jnp.cos(q_2) * jnp.sin(q_4) - jnp.cos(q_3) * jnp.cos(q_4) * jnp.sin(q_2)) + jnp.sin(
            q_2) * jnp.sin(q_3) * jnp.sin(q_5)) + 0.33
    pos_residual = np.array([x - x_next,y-y_next,z - z_next])
    return pos_residual
jac_constraint = jacobian(constraintfxn)



def costfxn(solverVariable,x_pos,goal):
  diff = solverVariable[7:] - goal
  v = value(x_pos)
  cost = np.matmul(diff.transpose(),np.matmul(v, diff ))
  w_des_vel = 0.002
  smoothness_cost = np.sum(solverVariable[0:7]**2,axis = 0)
  #print(f"cost = {cost}, smoothness = {smoothness_cost}")
  return np.add(cost , w_des_vel*smoothness_cost)
jac_cost = jacobian(costfxn)   #jaccost has a shape of (3,)


def trajMetricBased(init_joint,start_cart,end_cartesian):
    global t,q_prev,goal
    # print("Start : ",start_cart)
    # print("Goal : ", end_cartesian)
    goal = np.squeeze(end_cartesian)
    x_des  = end_cartesian
    num_dof = 7
    qdot = np.zeros(num_dof)  ########## initial velocity
    qdotprev = np.zeros(num_dof)
    q_min = np.array([-165, -100, -165, -165, -165, -1.0, -165]) * np.pi / 180
    #q_min = q_min.reshape(7,1)
    q_max = np.array([165, 101, 165, 1.0, 165, 214, 165]) * np.pi / 180
    #q_max = q_max.reshape(7,1)


    qdot_max = np.array([2.1750	,2.1750	,2.1750	,2.1750,	2.6100,	2.6100,	2.6100])
    qdot_min = -1*qdot_max
    qacc_max = np.array([15,	7.5,	10	,12.5	,15	,20,	20])
    qacc_min = -1*qacc_max

    x_pos = np.asarray(fk_franka(init_joint))  ########### initial positions

    q = init_joint
    q_prev = init_joint
      ######### delta t for which the computed velocity is commanded to the robot 3msec
    x_next = fk_franka(qdot*t + q)

    mpc_iter = 200

    q_1 = []
    q_2 = []
    q_3 = []
    q_4 = []
    q_5 = []
    q_6 = []
    q_7 = []
    cost_tracker = np.zeros(mpc_iter)
    trajectory = []
    qdottrajc =[]
    #pdb.set_trace()
    solverVariable = np.hstack((qdot,x_next))
    x_min = -np.ones(3,)*np.Inf #fk_franka(q_min)
    x_max = np.ones(3,)*np.Inf #fk_franka(q_max)
    solver_minbounds= np.hstack((qdot_min , x_min))
    solver_maxbounds = np.hstack((qdot_max , x_max))
    Amat = np.identity(10)
    Bmat = np.identity(10)
    for i in range(0, mpc_iter):
        # print(f"mpc-itr={i}")
        if np.linalg.norm(x_pos - x_des) < 0.01:
            break
            #t = 0.003
            # print("Without value velocity",v_des)

        bnds = Bounds(solver_minbounds,solver_maxbounds)
        nonlinear_constraint = NonlinearConstraint(constraintfxn , np.zeros((3,)), np.zeros((3,)) , jac= jac_constraint, hess=BFGS() , )

        linear_constraint_A = LinearConstraint(Amat*t, np.hstack((q_min-q_prev,x_min)), np.hstack((q_max-q_prev, x_max )))

        linear_constraint_B = LinearConstraint(Bmat ,np.hstack((qacc_min*t + qdotprev,x_min)), np.hstack((qacc_max*t + qdotprev,x_max)))
        defaultopts={ 'maxiter': 100, 'ftol': 1e-06, 'iprint': 1, 'disp': False, 'eps': 1.4901161193847656e-08, 'finite_diff_rel_step': None}
        res = minimize(costfxn ,  solverVariable, args =(x_pos,goal) , method='SLSQP', jac=jac_cost,
                constraints=[nonlinear_constraint,linear_constraint_A, linear_constraint_B], options=defaultopts ,bounds=bnds)  #TODO include linear constraints and smoothness cost
        #changing args for slsq might help ? like tol ??
        solverVariable = np.asarray(res['x']).squeeze()
        cost_tracker[i] = np.linalg.norm(np.hstack((x_pos[0] - x_des[0], x_pos[1] - x_des[1], x_pos[2] - x_des[2])))
        trajectory.append([q[0], q[1], q[2], q[3], q[4], q[5], q[6]])
        q = q + solverVariable[0:7] * t
        x_pos =  solverVariable[7:]
        qdotprev = solverVariable[0:7]
        q_prev = q
        qdottrajc.append(qdotprev)
    JointCost, CartCost = traj_cost(trajectory)
    return JointCost, CartCost, cost_tracker, np.array(trajectory)
