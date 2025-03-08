# -*- coding: utf-8 -*-
"""
author: Sylvain Bertrand, 2025

   All variables are in SI units
    
   
   Variables used by the functions of this script
    - t: time instant (s)
    - robotNo: no of the current robot for which control is coputed (0 .. nbRobots-1)
    - poses:  size (3 x nbRobots)
        eg. of use: the pose of robot 'robotNo' can be obtained by: poses[:,robotNo]
            poses[robotNo,0]: x-coordinate of robot position (in m)
            poses[robotNo,1]: y-coordinate of robot position (in m)
            poses[robotNo,2]: orientation angle of robot (in rad)   (in case of unicycle dynamics only)
"""


import numpy as np
import math
from lib.potential import Potential


# ==============   "GLOBAL" VARIABLES KNOWN BY ALL THE FUNCTIONS ==============
# all variables declared here will be known by functions below
# use keyword "global" inside a function if the variable needs to be modified by the function

# global toto

global firstCall   # can be used to check the first call ever of a function
firstCall = True

global pot # DO NOT MODIFY - allows initialisation of potential function from this script


# =============================================================================




# =============================================================================
def potential_seeking_ctrl(t, robotNo, robots_poses):
# =============================================================================

        
    # --- example of modification of global variables ---
    # ---(updated values of global variables will be known at next call of this funtion) ---
    # global toto
    # toto = toto +1
    
    global firstCall
    global pot
    


    # --- part to be run only once --- 
    if firstCall:
    
        # !!!!!!!!!!!!!!!!!!!!!!!  DO NOT REMOVE !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        #     YOU CAN MODIFY difficulty {1,2,3} AND random {True, False} PARAMETERS
        pot = Potential(difficulty=3, random=True)  
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        
        # you can add here other instructions to be executed only once
        
        firstCall = False
    # --------------------------------
    
    
    # get number of robots  (short notation)
    N = robots_poses.shape[0]
    
    # get index of current robot  (short notation)
    i = robotNo

    # get positions of all robots (short notation)
    x = robots_poses[:,0:2]

    # get potential values measured by all robots at their current positions (short notation)
    pot_measurement = np.zeros(N)
    for m in range(N):
        pot_measurement[m] = pot.value(x[m,:])

    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # to get access to potential measurement from robot i at time t in the rest of the code
    # you can use eihter use    pot_measurement[i]     or      pot.value(x[i,:])
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    
    
    
    # initialize control input vector for current robot i
    ui = np.zeros(2)
                       
    # compute control input for current robot i
    # ui = .... # <- TO BE COMPLETED 
    
    
    return ui[0], ui[1], pot   # potential is also returned to be used by main script for displays (DO NOT MODIFY)
# =============================================================================





# general template of a function defining a control law
# =============================================================================
def my_control_law(t, robotNo, robots_poses):
# =============================================================================  

    # --- example of modification of global variables ---
    # ---(updated values of global variables will be known at next call of this funtion) ---
    # global toto
    # toto = toto +1

    # number of robots
    nbOfRobots= robots_poses.shape[0]
    
    
    # control law
    vx = 0.
    vy = 0.

    # .................  TO BE COMPLETED HERE .............................
    
    return vx, vy
# =============================================================================

