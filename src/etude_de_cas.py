# -*- coding: utf-8 -*-
"""
author: Sylvain Bertrand, 2025


Simulation script of a Multi-Agent System, adapted for potential seeking

Agent dynamics can either be:
    -singleIntegrator2D:    x_dot = ux,   y_dot = uy
                            state : [x, y]
                            control input: [vx, vy]
    -unicycle:   x_dot = V.cos(theta),   y_dot = V.sin(theta),   theta_dot = omega
                            state : [x, y, theta]
                            control input: [V, omega] 
                            
    A conversion function is provided from single integrator inputs to unicycle inputs 
        [V, omega] = si_to_uni( [vx, vy], theta, kp=angular_speed_proportional_gain )

"""

import numpy as np
from lib.simulation import FleetSimulation
from lib.robot import Fleet, si_to_uni
import control_algo_potential
#from lib.potential import Potential
import matplotlib.pyplot as plt




# number of robots
# -----------------
nbOfRobots = 4


# dynamics of robots
# -------------------
robotDynamics = 'singleIntegrator2D'    # use 'singleIntegrator2D' or 'unicycle'


# initial states of robots 
# --------------------------

# ... initial positions randomly defined 
#initPositions = 40*np.random.rand(nbOfRobots,2)-20  # random init btw -20, +20

# ... initial positions defined from data      (dimension: nb of agents  x  2)
initPositions = np.array([[ -20, -21, -21, -20 ],       # x-coordinates (m)
                          [-20, -20, -21, -21 ]]).T   # y-coordinates (m)


# ... initial orientation angles and poses (USED FOR UNICYCLE DYNAMICS ONLY)
if (robotDynamics=='unicycle'):
    # orientation angles (rad)      (dimension: nb of agents x 1)
    initAngles = np.array([[0., 0., 0., 0.]]).T   
    initPoses = np.concatenate((initPositions, initAngles), axis=1)



# create fleet
if (robotDynamics=='singleIntegrator2D'):
    fleet = Fleet(nbOfRobots, dynamics=robotDynamics, initStates=initPositions)
else:
    fleet = Fleet(nbOfRobots, dynamics=robotDynamics, initStates=initPoses)


# sampling period for simulation
Ts = 0.05

# create simulation
simulation = FleetSimulation(fleet, t0=0.0, tf=6.0, dt=Ts)

# create history of potential measurements done by the robots
potential_measurements = np.zeros((simulation.t.shape[0],nbOfRobots))
t_index=0

   

# simulation loop
for t in simulation.t:

    # get poses of robots as a single array of size (nb_of_robots, nb_of_states)
    robots_poses = fleet.getPosesArray()    

    # compute control input of each robot
    for robotNo in range(fleet.nbOfRobots):
        
    
        vx, vy, pot = control_algo_potential.potential_seeking_ctrl(t, robotNo, robots_poses)              # <= MODIFY CONTROL LAW AND POTENTIAL DEFINITION IN "control_algo_potential.py"
        
        
        if (robotDynamics=='singleIntegrator2D'):
            fleet.robot[robotNo].ctrl = np.array([vx, vy]) 
        else:
            fleet.robot[robotNo].ctrl = si_to_uni(vx, vy, robots_poses[robotNo,2], kp=10.) 


    # display potential values measured by the robots and maximum value to be found
    if (t_index==0):
        print( '\n[      potential value measured by each robot       ] | max value to be found')
        print( '------------------------------------------------------------------------------')
    else:
        print( str(pot.value(robots_poses[:,0:2])) +  ' | ' + str(np.max(pot.value(pot.mu))))
        

    # store potential measurements in history (for plots)
    potential_measurements[t_index,:] = pot.value(robots_poses[:,0:2])
    if (-10 in potential_measurements[t_index,:]):
        print("!!!!!!!!!!!!! ERROR in potential value definition !!!!!!!!!!!!!")
        print('Please modify Potential parameters or start another simulation if using random definition')
    t_index += 1
    
    
    # update data of simulation 
    simulation.addDataFromFleet(fleet)

    # integrate motion of the fleet
    fleet.integrateMotion(Ts)




# plot animation (press [ESC] to abort and close simulation window)
'''
fig1, ax1 = pot.plot(1)
simulation.animation(figNo=1, potential=pot, pause=0.001, robot_scale=0.2, xmin=-25, xmax=25, ymin=-25, ymax=25)   
'''

# plot 2D trajectories
simulation.plotXY(figNo=2, potential=pot, xmin=-25, xmax=25, ymin=-25, ymax=25)

# plot states' components Vs time
simulation.plotState(figNo=3)

# plot control inputs' components Vs time
simulation.plotCtrl(figNo=6)

# plot 2D trajectories (every 'X steps' time instants
#simulation.plotXY(figNo=10, steps=50, links=True)



# plot time history of potential measurements done by the robots and maximum value to be found
plt.figure()
colorList = ['r', 'g', 'b', 'y', 'c', 'm', 'k']
for rr in range(fleet.nbOfRobots):
    plt.plot(simulation.t, potential_measurements[:,rr], color=colorList[rr])
plt.plot(simulation.t, np.max(pot.value(pot.mu))*np.ones_like(simulation.t), 'k:')
plt.text(0, np.max(pot.value(pot.mu))*1.01, 'max value to be found')
plt.xlabel('t (s)')
plt.ylabel('Potential value (-)')
plt.grid()


