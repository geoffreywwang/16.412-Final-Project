import numpy as np
from math import erf
# from generate_objects import generate_objects
from scipy.special import erfinv

from RMPC_robot import RMPC, LCC, Object
# from utils import *
import pulp as pl
# import ira
from visualize_environment import visualize

from ira import IRA



class Planner():
    def __init__(self, T, xdim, udim, T_length, A, B, x0, xf, Sigma_x0, Sigma_w, Delta, state_bounds, input_bounds):
        self.T = T # timesteps
        self.T_length = T_length # Time length for each timestep
        self.A = A
        self.B = B
        self.x0_bar = x0
        self.xf_bar = xf
        self.Sigma_x0 = Sigma_x0
        self.Sigma_w = Sigma_w
        self.xdim = xdim
        self.udim = udim
        self.state_bounds = state_bounds
        self.input_bounds = input_bounds
        self.Delta = Delta
        self.rmpc = RMPC(self.T, self.xdim, self.state_bounds, self.udim, self.input_bounds, self.A, self.B, self.x0_bar, self.Sigma_x0, self.Sigma_w, self.Delta, self.xf_bar)
        self.object_means = []
        self.object_covs = []
        self.objects = []

    def include_trajectories(self, trajectories):
        N, T = trajectories.shape[0], trajectories.shape[1]
        for t in range(T):
            self.object_means.append(np.mean(trajectories[:, t, :], axis=0))
            cov = np.zeros((4, 4))
            cov[:2, :2] = np.cov(trajectories[:, t, :].T)
            self.object_covs.append(cov)
        
        self.rmpc.update_object_sigmas(self.object_covs)
        for i, pos in enumerate(self.object_means):
            self.objects.append(Object(pos[0], pos[1], .3, .3, [i]))

        for obj in self.objects:    
            self.rmpc.add_object(obj, None)


    def update_delta(self, new_Delta):
        self.Delta = new_Delta
        self.rmpc = RMPC(self.T, self.xdim, self.state_bounds, self.udim, self.input_bounds, self.A, self.B, self.x0_bar, self.Sigma_x0, self.Sigma_w, self.Delta, self.xf_bar)

    def reset_rmpc(self):
        self.rmpc = RMPC(self.T, self.xdim, self.state_bounds, self.udim, self.input_bounds, self.A, self.B, self.x0_bar, self.Sigma_x0, self.Sigma_w, self.Delta, self.xf_bar)
        self.object_means = []
        self.object_covs = []
        self.objects = []


    def plan_with_IRA(self, alpha=0.7, active_tol=0.001, convergence_tol=0.001):
        IRA(self.rmpc, alpha, active_tol, convergence_tol)

    def plan_without_IRA(self):
        for i in range(len(self.rmpc.lcc)):
            self.rmpc.lcc[i].delta = self.rmpc.Delta/len(self.rmpc.lcc)
        self.rmpc.determinize_and_solve()

    def get_x(self):
        return [self.rmpc.xbar(i) for i in range(self.T + 1)]

    def get_u(self):
        return [self.rmpc.u(i) for i in range(self.T)]

    def get_M(self):
        return [self.rmpc.M(i) for i in range(self.T)]

    def get_objects(self):
        return self.objects

    def get_objective(self):
        return self.rmpc.objective()





