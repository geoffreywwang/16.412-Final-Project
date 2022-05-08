from time import time
from pulp import LpProblem, LpMinimize, LpVariable, lpDot, lpSum, value, PULP_CBC_CMD, LpBinary
# from numpy import dot, empty, sqrt, array, eye
import numpy as np
from numpy.linalg import inv
from scipy.special import erfinv


class Object:
    def __init__(self, x, y, w, h, ts):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.timesteps = ts

class LCC:
    """
    Represents a Linear Chance Constraint (LCC), or namely a chance constraint
    defined over a linear inequality.
    """

    def __init__(self, k, h, g, delta, use_milp=False, M_i=None, M_j=None):
        self.k = k
        self.h = h
        self.g = g
        self.delta = delta
        self.use_milp = use_milp
        self.M_i = M_i
        self.M_j = M_j


class RMPC:
    """
    A class that can represent a robust model-predicitve control problem.
    It can also convert it to a determinized version.
    """

    def __init__(self, timesteps, xdim, xlims, udim, ulims, A, B, x0_bar, Sigma_x0, Sigma_w, Delta, xf_bar):
        """
        Initializes the class. Takes in the following inputs:
        
            timesteps - an integer number of timesteps (such as 5)

            xdim - the dimensionality of the state vector x. It will be stored as an xdim column vector.

            xlims - a list of limits on x, the problems state variables. Each item in the list
                    should be a tuple of ranges, like (20, 30), which means that 20 <= x_i <= 30
                    across all timesteps. If xlims contains just a single tuple, it is applied to all
                    xdim of the state variables in x. If xlims contains xdim tuples, then each is applied
                    individually. Anything else results in an error.

            udim - the dimensionality of the control vector u.

            ulims - same idea as xlims, but for u instead of x.

            A, B - the matrices that specify the discrete-time dynamics of the system, which take the form
                   x_{k+1} = A x_{k} + B u_{k}. Specified as NumPy matrices. `A` should be xdim x xdim in 
                   size, and `B` should be xdim x udim.

            x0_bar, Sigma_x0 - specify the initial distribuition of x at timestep k=0.
                               x is assumed to be normally distributed about N(x0_bar, Sigma_x0).
                               x0_bar should be a vector of length xdim, and Sigma_x0 should be
                               an xdim x xdim matrix.

            Sigma_w - the covariance matrix of the additive noise w_k (which is assumed to have zero mean).
                      This should be an xdim x xdim matrix.

            Delta - the overall chance constraint.
        """
        # We may either have one set of limits for all indices or a set of limits for each index
        if len(xlims) != 1 and len(xlims) != xdim:
            raise Exception('Incorrectly specified x limits.')
        if len(ulims) != 1 and len(ulims) != udim:
            raise Exception('Incorrectly specified u limits.')

        # Create a linear program to be solved
        self._problem = LpProblem("RMPC_Problem", LpMinimize)

        # A list of linear chance constraints (lcc's)
        self.lcc = []

        # Arrays/matrices to hold variables and covariances
        self._xbar = np.empty((timesteps + 1, xdim), dtype=LpVariable)
        self._u = np.empty((timesteps, udim), dtype=LpVariable)
        self.abs_u = np.empty((timesteps, udim), dtype=LpVariable)
        self._Sigma_x = np.empty((timesteps + 1), dtype=object)
        self._M = np.empty((timesteps + 1, 4), dtype=LpVariable)
        self._Object_Sigmas = [np.zeros((xdim, xdim)) for i in range(timesteps + 1)]

        # Make all state variables
        for k in range(timesteps + 1):
            for i in range(xdim):
                if len(xlims) == 1:
                    self._xbar[k, i] = LpVariable('x_{},{}'.format(k, i), xlims[0][0], xlims[0][1])
                else:
                    self._xbar[k, i] = LpVariable('x_{},{}'.format(k, i), xlims[i][0], xlims[i][1])
        
        # Make all M variables
        for k in range(timesteps + 1):  
            for i in range(4):  
                self._M[k, i] = LpVariable("M_{}_{}".format(k, i), cat=LpBinary)

        # Make all control variables
        for k in range(timesteps):
            for i in range(udim):
                if len(ulims) == 1:
                    self._u[k, i] = LpVariable('u_{},{}'.format(k, i), ulims[0][0], ulims[0][1])
                else:
                    self._u[k, i] = LpVariable('u_{},{}'.format(k, i), ulims[i][0],ulims[i][1])

        # Make variables for the absolute value of control variables
        for k in range(timesteps):
            for i in range(udim):
                if len(ulims) == 1:
                    self.abs_u[k, i] = LpVariable('|u|_{},{}'.format(k, i), 0, 
                                                 max(abs(ulims[0][0]), abs(ulims[0][1])))
                else:
                    self.abs_u[k, i] = LpVariable('|u|_{},{}'.format(k, i), 0, 
                                                 max(abs(ulims[i][0]), abs(ulims[i][1])))

        # Store RMPC problem parameters
        self.timesteps = timesteps
        self.A = A
        self.B = B
        self.x0_bar = x0_bar
        self.Sigma_x0 = Sigma_x0
        self.Sigma_w = Sigma_w
        self.Delta = Delta
        self.xf_bar = xf_bar
        self.M_val = 1000

        # Store the solved / unsolved (or no solution) status.
        self.status = -1 # Not solved

    def add_lcc(self, k, h, g, delta):
        """Create and store a linear chance constraint."""
        self.lcc.append(LCC(k, h, g, delta))

    def add_object(self, obj, delta):
        h_x = np.array([1, 0, 0, 0])
        h_y = np.array([0, 1, 0, 0])
        x_lb = obj.x - obj.w/2
        x_ub = obj.x + obj.w/2
        y_lb = obj.y - obj.h/2
        y_ub = obj.y + obj.h/2
        for k in obj.timesteps:
            self.lcc.append(LCC(k,  h_x,  x_lb, delta, True, k, 0))
            self.lcc.append(LCC(k, -h_x, -x_ub, delta, True, k, 1))
            self.lcc.append(LCC(k,  h_y,  y_lb, delta, True, k, 2))
            self.lcc.append(LCC(k, -h_y, -y_ub, delta, True, k, 3))

    def _compute_variance(self):
        """
        Compute variances at all timesteps using problem parameters. Note that these
        can be computed in advance (as is done here), since the variance will continue to grow
        and doesn't depend at all on the control variable values, or any other values.
        """
        self._Sigma_x[0] = self.Sigma_x0
        for k in range(self.timesteps):
            self._Sigma_x[k+1] = np.dot(np.dot(self.A, self._Sigma_x[k]), self.A.transpose()) + self.Sigma_w
        # add object sigmas, representing a convolution of uncertainties
        for k in range(self.timesteps + 1):
            self._Sigma_x[k] = self._Sigma_x[k] + self._Object_Sigmas[k] 

    def update_object_sigmas(self, object_sigmas):
        for i, object_sigma in enumerate(object_sigmas):
            self._Object_Sigmas[i] = object_sigma

    def _encode_linear_dynamics(self):
        """Encode linear dynamic constraints."""
        (rA, cA) = self.A.shape
        (rB, cB) = self.B.shape
        (rx, cx) = self._xbar.shape
        (ru, cu) = self._u.shape

        if cA != cx:
            raise Exception('Shape of A inconsistent with x_(k).')
        if cB != cu:
            raise Exception('Shape of B inconsistent with u_(k).')
        if rA != rB:
            raise Exception('Shape of A inconsistent with shape of B.')

        # x_{k+1} = A*x_{k} + B*u_{k}
        for k in range(ru):
            for i in range(rA):
                self._problem += (self._xbar[k+1, i]
                                  == lpDot(self.A[i, :].tolist(), self._xbar[k, :].tolist())
                                     + lpDot(self._u[k, :].tolist(), self.B[i, :].tolist()))

        # Initial state
        for i in range(rA):
            self._problem += (self._xbar[0, i] == self.x0_bar[i])
        # Final state
        for i in range(rA):
            self._problem += (self._xbar[self.timesteps-1, i] == self.xf_bar[i])
        # Checkpoint
        # cpt = [65, 60]
        # for i in range(rA//2):
        #     self._problem += (self._xbar[self.timesteps//2, i] == cpt[i])

        # A trick to compute the absolute value of control with linear inequalities
        for k in range(ru):
            for i in range(cu):
                self._problem += (self.abs_u[k, i] >= self._u[k, i])
                self._problem += (self.abs_u[k, i] >= -1*self._u[k, i])

    def _encode_probabilistic_constraints(self):
        """Encode chance constraints as deterministic constraints."""
        for con in self.lcc:
            if con.delta == None:
                raise Exception('Missing risk allocation on probabilistic linear constraint.')
            if not con.use_milp:
                self._problem += (lpDot(con.h.tolist(), self._xbar[con.k,:].tolist())
                                <= con.g - np.sqrt(2*np.dot(np.dot(con.h, self._Sigma_x[con.k]), con.h)) * erfinv(1 - 2*con.delta))
            else:
                # print("h: ", con.h, " g: ", con.g, " M_j: ", con.M_j, " k: ", con.k)
                self._problem += (lpDot(con.h.tolist(), self._xbar[con.k,:].tolist())
                                <= con.g - np.sqrt(2*np.dot(np.dot(con.h, self._Sigma_x[con.k]), con.h)) * erfinv(1 - 2*con.delta) + self._M[con.k, con.M_j]*self.M_val)  

    def _encode_objective_function(self):
        """Encode the objective: minimizing the sum of the absolute values of all control variables."""
        self._problem += lpSum(self.abs_u.flatten().tolist())

    def _encode_M_constraint(self):
        for k in range(self.timesteps + 1):
            self._problem += 3 == lpSum(self._M[k, :].flatten().tolist())

    def _solve(self):
        """Solve the problem, and raise an exception if it is unsolvable."""
        self.status = self._problem.solve(PULP_CBC_CMD(msg=False))
        if self.status == -1:
            raise Exception('No solution found.')

    def determinize_and_solve(self):
        """
        This method encodes all of the constraints, determinizes and then encodes all of the
        probabilistic constraints (i.e., the LCC's), and then solves the determinized system.
        """
        self._problem = LpProblem("RMPC_Problem", LpMinimize)
        self._compute_variance()
        self._encode_linear_dynamics()
        self._encode_probabilistic_constraints()
        self._encode_objective_function()
        self._encode_M_constraint()
        self._solve()

    def objective(self):
        """
        Returns the objective value of the solved problem, 
        or an exception if it hasn't been solved yet.
        """
        if self.status == -1:
            raise Exception('Please call determinize_and_solve() successfully before calling objective()!')
        return value(self._problem.objective)

    def xbar(self, k):
        """
        Returns the solved value for xbar (the mean of x) at timestep k.
            Input:
                k - an integer from 0 to timesteps
        Throws an error if the system hasn't been solved successfully yet.
        """
        if self.status == -1:
            raise Exception('Please call determinize_and_solve() successfully before calling xbar()!')
        return np.array([value(xbar_i) for xbar_i in self._xbar[k, :]])

    def Sigma_x(self, k):
        """
        Returns the value for Sigma_x (the covariance matrix of x) at timestep k.
            Input:
                k - an integer from 0 to timesteps
        Throws an error if the system hasn't been solved successfully yet.
        """
        if self.status == -1:
            raise Exception('Please call determinize_and_solve() successfully before calling Sigma_x()!')
        return self._Sigma_x[k]

    def u(self, k):
        """
        Returns the solved value for u (the control value) at timestep k.
            Input:
                k - an integer from 0 to timesteps - 1
        Throws an error if the system hasn't been solved successfully yet.
        """
        if self.status == -1:
            raise Exception('Please call determinize_and_solve() successfully before calling u()!')
        return np.array([value(ui) for ui in self._u[k, :]])

    def M(self, k):
        if self.status == -1:
            raise Exception('Please call determinize_and_solve() successfully before calling M()!') 
        # print(self._M[0, 0])
        return np.array([value(m) for m in self._M[k, :]])
