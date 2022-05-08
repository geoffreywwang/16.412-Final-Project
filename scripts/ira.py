import numpy as np
from numpy import array, dot, sqrt, inf
from math import erf
from scipy.special import erfinv


from RMPC_robot import RMPC, LCC
# from utils import *
import pulp as pl


def initialize_risk_allocation(rmpc):
    """
    Initialize the risk allocation over all linear chance constraints.

    This function should iterate over all lcc's stored in the RMPC, and set their
    initial risk allocation values (i.e., lcc.delta = ...).

    Input:
        rmpc - a robust model-predictive control problem instance

    Returns: (nothing)
    """
    for i in range(len(rmpc.lcc)):
        rmpc.lcc[i].delta = rmpc.Delta/len(rmpc.lcc)

def probability_of_linear_constraint_holding(h, g, xbar, Sigma_x):
    """
    Computes the probability that the constaint h^T x <= g holds true, given
    that x is a random variable with a multivariate normal distribution of
    mean xbar and covariance Sigma_x.

    Input:
        h - a column vector
        g - a scalar
        xbar - a column vector specifying the mean of x
        Sigma_x - a matrix specifying the covariance of x

    Returns:
        The probability that h^T x <= g holds.
    """
    num = g - h.T.dot(xbar)
    den = np.sqrt(2*np.dot(h.T.dot(Sigma_x), h))
    return .5 + .5*erf(num/den)

def is_linear_chance_constraint_active(rmpc, lcc, eta=0.001):
    """
    Determines if the given lcc is active.

    Input: 
        rmpc - a robust model-predictive control problem instance
        lcc - a linear chance constraint
        eta - the tolerance eta for checking if the constraint is active.

    Returns: True/False
    """
    k = lcc.k
    xbar = rmpc.xbar(k)
    Sigma_x = rmpc.Sigma_x(k)
    g = lcc.g
    h = lcc.h
    allowed = 1 - lcc.delta
    M_j = lcc.M_j
    # print("M: ", rmpc.M(k)[M_j])
    if rmpc.M(k)[M_j] == 1:
        return True 
    # print("true: ", probability_of_linear_constraint_holding(h, g, xbar, Sigma_x), "allowed: ", allowed)
    return abs(probability_of_linear_constraint_holding(h, g, xbar, Sigma_x) - allowed) <= eta

def IRA(rmpc, alpha=0.7, active_tol=0.001, convergence_tol=0.001):
    # YOUR CODE HERE
    J = np.inf
    initialize_risk_allocation(rmpc)
    J_new = 0
    n_con = len(rmpc.lcc)
    count = 0
    while abs(J - J_new) > convergence_tol:
        # print("IRA STEP", count)
        count += 1
        # for con in rmpc.lcc:
        #     print(con.delta)
        J = J_new
        # print("HI")
        rmpc.determinize_and_solve()
        J_new = rmpc.objective()
        active = []
        non_active = []
        for con in rmpc.lcc:
            if is_linear_chance_constraint_active(rmpc, con, eta=0.001):
                active.append(con)
            else:
                non_active.append(con)
        # print(len(active))
        if len(active) == n_con or len(active) == 0:
            break
        for con in non_active:
            k = con.k
            xbar = rmpc.xbar(k)
            Sigma_x = rmpc.Sigma_x(k)
            g = con.g
            h = con.h
            cdf = .5 + .5*erf(g - np.dot(h.T, xbar))
            # print("prob holding: ", probability_of_linear_constraint_holding(h, g, xbar, Sigma_x))
            con.delta = alpha*con.delta + (1 - alpha)*(1 - probability_of_linear_constraint_holding(h, g, xbar, Sigma_x))
        d_residual = rmpc.Delta - sum([con.delta for con in rmpc.lcc])
        # print(d_residual)
        for con in active:
            con.delta = con.delta + d_residual/len(active)
