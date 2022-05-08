from planner import Planner
import numpy as np
from visualize_environment import visualize

###### GENERATE ROBOT DYNAMICS ############
T = 20 # timesteps
T_length = .5 # length of each timestep
A = np.identity(4)
A[0, 2] = 1*T_length
A[1, 3] = 1*T_length
B = np.zeros((4, 2))
B[2, 0] = 1*T_length
B[3, 1] = 1*T_length

x0_bar = np.array([0, 0, 0, 0]) # The mean of x at timestep 0
Sigma_x0 = np.zeros((4, 4)) # Covariance matrix of x at timestep 0
Sigma_x0[0, 0] = .05
Sigma_x0[1, 1] = .05
Sigma_x0[2, 2] = .001
Sigma_x0[3, 3] = .001
Sigma_w = 0.00001*np.identity(4) # Noise covariance added to x at each timestep
xf_bar = np.array([10, 10, 0, 0]) # Final state
state_bounds = [(-40, 40), (-40, 40), (-15, 15), (-15, 15)] # Bounds for our state vector
input_bounds = [(-1.2, 1.2), (-1.2, 1.2)] # Bounds for our input vector

##### GENERATE DUMMY TRAJECORIES ###########
N = 30 # Num trajectories
data = np.zeros((N, T + 1, 2)) # array for trajectories
start = (2, 8) # Start point on dummy trajectories
travel = (7, -7) # x and y movement along each direction
# Genearte trajectories
for t in range(T + 1):
    for n in range(N):
        mean = np.array([start[0] + travel[0]*t/T, start[1] + travel[1]*t/T])
        cov = np.array([[1, 0], [0, 1]])
        data[n, t, :] = np.random.multivariate_normal(mean, cov)

######### EX 1: PLAN WITHOUT OBJECTS #################
Delta = 0.1 # Set chance constraint
p = Planner(T, 4, 2, T_length, A, B, x0_bar, xf_bar, Sigma_x0, Sigma_w, Delta, state_bounds, input_bounds)
p.plan_with_IRA() # optimize path
x1 = p.get_x() # get plan states
print("Objective: ", p.get_objective())
visualize(xs=[x1], labels=["No obstacle plan"], objs=[], c=["r"])
print()

######### EX 2: PLAN WITH OBJECTS #####################
Delta = 0.1 # Set chance constraint
p = Planner(T, 4, 2, T_length, A, B, x0_bar, xf_bar, Sigma_x0, Sigma_w, Delta, state_bounds, input_bounds)
p.include_trajectories(data) # Add trajectories to planner
p.plan_with_IRA() # optimize path
x1 = p.get_x() # get plan states
print("Objective: ", p.get_objective())
# visualize
objs = p.get_objects()
visualize(xs=[x1], labels=["Plan with obstacles"], objs=objs, c=["r"])
print()

######### EX 3: COMPARE DIFFERENT DELTAS ##############
# Plan with Delta = .1
Delta = 0.1 # Set chance constraint
p1 = Planner(T, 4, 2, T_length, A, B, x0_bar, xf_bar, Sigma_x0, Sigma_w, Delta, state_bounds, input_bounds)
p1.include_trajectories(data) # Add trajectories to planner
p1.plan_with_IRA() # optimize path
x1 = p1.get_x() # get plan states
# Plan with Delta = .4
Delta = 0.4 # Set chance constraint
p2 = Planner(T, 4, 2, T_length, A, B, x0_bar, xf_bar, Sigma_x0, Sigma_w, Delta, state_bounds, input_bounds)
p2.include_trajectories(data) # Add trajectories to planner
p2.plan_with_IRA() # optimize path
x2 = p2.get_x() # get plan states
# Get objective value
print("Delta=.1 Objective: ", p1.get_objective())
print("Delta=.4 Objective: ", p2.get_objective())
# Get average distance between both plans
total_distance = 0
for i in range(len(x1)):
    total_distance += np.linalg.norm(x2[i] - x1[i])
avg_distance = total_distance/len(x1)
print("Avg distance between plans: ", avg_distance)
# visualize
objs = p2.get_objects()
visualize(xs=[x1, x2], labels=["Delta = .1", "Delta = .4"], objs=objs, c=["r", "g"])
print()

######## EX 4: COMPARE IRA VS VANILLA OPTIMIZATION ####
# Plan with IRA
Delta = 0.4 # Set chance constraint
p1 = Planner(T, 4, 2, T_length, A, B, x0_bar, xf_bar, Sigma_x0, Sigma_w, Delta, state_bounds, input_bounds)
p1.include_trajectories(data) # Add trajectories to planner
p1.plan_with_IRA() # optimize path with IRA
x1 = p1.get_x() # get plan states
# PLan without IRA
Delta = 0.4 # Set chance constraint
p2 = Planner(T, 4, 2, T_length, A, B, x0_bar, xf_bar, Sigma_x0, Sigma_w, Delta, state_bounds, input_bounds)
p2.include_trajectories(data) # Add trajectories to planner
p2.plan_without_IRA() # optimize path without IRA
x2 = p2.get_x() # get plan states
# Get objective value
print("IRA Objective: ", p1.get_objective())
print("No IRA Objective: ", p2.get_objective())
# Get average distance between both plans
total_distance = 0
for i in range(len(x1)):
    total_distance += np.linalg.norm(x2[i] - x1[i])
avg_distance = total_distance/len(x1)
print("Avg distance between plans: ", avg_distance)
# visualize
objs = p2.get_objects()
visualize(xs=[x1, x2], labels=["IRA", "No IRA"], objs=objs, c=["r", "g"])