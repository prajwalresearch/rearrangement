import roboticstoolbox as rtb
import cvxopt
import autograd.numpy as np
import pdb
import cvxopt
from cvxopt import solvers
import random
import matplotlib.pyplot as plt
from spatialmath import SE3
from autograd import grad
import csv
import MetricBased
robot = rtb.models.DH.Panda()



def traj_cost(trajectory):
    cost = 0
    cart_cost = 0
    for i in range(len(trajectory) - 1):
        cost += np.linalg.norm(np.asarray(trajectory[i]) - np.asarray(trajectory[i + 1]), ord=2)
        # pdb.set_trace()
        current = robot.fkine(trajectory[i]).A
        next = robot.fkine(trajectory[i + 1]).A
        cart_cost += np.linalg.norm(current[:3, -1] - next[:3, -1], ord=2)
    return cost, cart_cost


def findTrajectoryIK(init_joint, x_start , x_des, USeValuefunction=False):
    num_dof = 7
    qdot = np.zeros(num_dof)  ########## initial velocity
    q_min = np.array([-165, -100, -165, -165, -165, -1.0, -165]) * np.pi / 180
    q_max = np.array([165, 101, 165, 1.0, 165, 214, 165]) * np.pi / 180
    # q = (q_min + q_max) / 2.0  ########## initial joint angle
    x_pos = robot.fkine(init_joint).A  ########### initial positions
    x_pos = x_pos[:3, -1]
    # x_pos = init_joint
    w_des_vel = 0.001
    #pdb.set_trace()

    q = init_joint
    t = 0.02  ######### delta t for which the computed velocity is commanded to the robot 3msec
    if USeValuefunction:
        mpc_iter =7000
    else:
        mpc_iter =20000
    qdot_max = np.ones(num_dof) * 2.1

    q_1 = []
    q_2 = []
    q_3 = []
    q_4 = []
    q_5 = []
    q_6 = []
    q_7 = []
    Optcost= []
    cost_tracker = np.zeros(mpc_iter)
    trajectory = []
    Jacfranka = []

    for i in range(0, mpc_iter):
        #print((f"USeValuefunction=", {i}) if USeValuefunction else (f"WithoutUSeValuefunction=", {i}))
        # Optcost.append(value(x_pos,x_des))
        if np.linalg.norm(x_pos - x_des) < 0.01:
            # pdb.set_trace()
            if USeValuefunction:
                print("Value Goal Reached in iterations : ", i)
            else:
                print("Without Value Goal Reached in iterations : ", i)
            break

        if USeValuefunction:
            w_smoothness = 0.001
            # v_des = -grad_value(x_pos)[:3]
            k_p = 0.1
            #t = 0.0008
        else:
            w_smoothness = 0.001
            v_des = (x_des - x_pos)
            k_p = 0.1
            #t = 0.003

            # print("Without value velocity",v_des)

        #if np.linalg.norm(v_des) > 1:
        #    v_des = v_des / np.linalg.norm(v_des)
        #     print(np.concatenate((x_des[:2], x_pos[:2], x_des[2]- x_pos), axis=0))
        Jac_franka = robot.jacob0(q)[0:3, :]
        #Jacfranka.append(np.linalg.det(np.matmul(Jac_franka.transpose(), Jac_franka )))
        #pdb.set_trace()
        ############ cost function minimizes the goal reaching cost and smoothness cost modeled as finite difference of velocity
        cost = w_des_vel * np.dot(Jac_franka.T, Jac_franka) + w_smoothness * np.identity(num_dof)

        lincost = -w_des_vel * np.dot(Jac_franka.T, k_p * v_des) - w_smoothness * qdot

        A_ineq = np.vstack(
            (np.identity(num_dof), -np.identity(num_dof), np.identity(num_dof) * t, -np.identity(num_dof) * t))
        b_ineq = np.hstack((qdot_max, qdot_max, q_max - q, -q_min + q))

        sol = solvers.qp(cvxopt.matrix(cost, tc='d'), cvxopt.matrix(lincost, tc='d'), cvxopt.matrix(A_ineq, tc='d'),
                         cvxopt.matrix(b_ineq, tc='d'), None, None)

        qdot = np.asarray(sol['x']).squeeze()  ########### joint velocity to be given to Py-bullet or ROS

        cost_tracker[i] = np.linalg.norm(np.hstack((x_pos[0] - x_des[0], x_pos[1] - x_des[1], x_pos[2] - x_des[2])))
        trajectory.append([q[0], q[1], q[2], q[3], q[4], q[5], q[6]])

        q = q + qdot * t
        pos = robot.fkine([q[0], q[1], q[2], q[3], q[4], q[5], q[6]]).A
        x_pos[0] = pos[0, -1]
        x_pos[1] = pos[1, -1]
        x_pos[2] = pos[2, -1]
    JointCost, CartCost = traj_cost(trajectory)
    #print((f"USeValuefunction=", {i}) if USeValuefunction else (f"WithoutUSeValuefunction=", {i}))
    return JointCost, CartCost, cost_tracker, np.array(trajectory)
