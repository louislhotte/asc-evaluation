# -*- coding: utf-8 -*-
"""
Robot Class

Author: S. Bertrand, 2023
"""

import math
import numpy as np
import matplotlib.pyplot as plt



# converts cartesian velocity input into linear and angular speed input, at a given orientation theta 
# -----------------------------------------------------------------------------
def si_to_uni(vx, vy, theta, kp=1.0):
# -----------------------------------------------------------------------------   
    V = np.sqrt(vx**2 + vy**2)
    
    theta_ref = np.arctan2(vy, vx)
    # to avoid U-turns
    if math.fabs(theta-theta_ref)>math.pi:
            theta_ref += math.copysign(2*math.pi, theta)
    
    omega = kp*(theta_ref - theta)
    
    return V, omega






# =============================================================================
class Robot:
# =============================================================================
    
    # -------------------------------------------------------------------------
    def __init__(self, dynamics='singleIntegrator2D', robotNo=0, initState=[]):
    # -------------------------------------------------------------------------
    
        self.robotNo = robotNo
        self.dynamics = dynamics
        
        
        if (dynamics == 'singleIntegrator2D'):
            # state
            self.stateDim = 2
            if (len(initState)==0):
                self.state = np.array([0., 0.]) # x,y
            else:
                self.state = initState
            # control
            self.ctrlDim = 2
            self.ctrl = np.array([0.,0.]) # vx, vy
            
            
        if (dynamics == 'unicycle'):
            # state
            self.stateDim = 3
            if (len(initState)==0):
                self.state = np.array([0., 0., 0.]) #x, y, theta
            else:
                self.state = initState
            # control
            self.ctrlDim = 2
            self.ctrl = np.array([0.,0.]) # V, omega
        

    # -------------------------------------------------------------------------
    def setCtrl(self, ctrl):
    # -------------------------------------------------------------------------
        self.ctrl = ctrl      
            
            
    
    # -------------------------------------------------------------------------
    def integrateMotion(self, Te):        
    # -------------------------------------------------------------------------
    # integrate robot motion over one sampling period (Euler discretization) applying control input u    
    
        if (self.dynamics == 'singleIntegrator2D'):          
            self.state = self.state + Te * self.ctrl
            
        if (self.dynamics == 'unicycle'):
            self.state[0] = self.state[0] + Te * self.ctrl[0]*np.cos(self.state[2])
            self.state[1] = self.state[1] + Te * self.ctrl[0]*np.sin(self.state[2])
            self.state[2] = self.state[2] + Te * self.ctrl[1]
            
  
        
    # -------------------------------------------------------------------------        
    def __repr__(self):
    # -------------------------------------------------------------------------
        """Display in command line"""
        message = "\nRobot:\n index: {}\n".format(self.index)
        message += " state: {}".format(self.state)        
        return message+"\n"
    
    # -------------------------------------------------------------------------
    def __str__(self):
    # -------------------------------------------------------------------------
        """Display with print function"""
        message = "\nRobot:\n no: {}\n".format(self.robotNo)
        message += " state: {}".format(self.state)        
        return message+"\n"

# ====================== end of class Robot ===================================




# =============================================================================
class Fleet:
# =============================================================================
    
    # -------------------------------------------------------------------------
    def __init__(self, nbOfRobots, dynamics='singleIntegrator2D', initStates=[]):
    # -------------------------------------------------------------------------
        self.nbOfRobots = nbOfRobots
        
        self.robot = []
        
        for robotNo  in range(self.nbOfRobots):
            if (len(initStates)>0):
                self.robot.append( Robot(dynamics, robotNo, initStates[robotNo,:]) )
            else:
                self.robot.append( Robot(dynamics, robotNo, initStates) )

        
    # -------------------------------------------------------------------------
    def integrateMotion(self, Te):        
    # -------------------------------------------------------------------------
    # integrate fleet motion over one sampling period (Euler discretization) applying control input u    
    
        for i in range(self.nbOfRobots):    
            self.robot[i].integrateMotion(Te)
    
        
    # -------------------------------------------------------------------------        
    def __repr__(self):
    # -------------------------------------------------------------------------
        """Display in command line"""
        message = "\nFleet\n"
        for rob in self.robot:
            message += " Robot:\n no: {}\n".format(rob.robotNo)
            message += "  state: {}\n".format(rob.state)        
        return message+"\n"
    
    # -------------------------------------------------------------------------
    def __str__(self):
    # -------------------------------------------------------------------------
        """Display with print function"""
        message = "\nFleet\n"
        for rob in self.robot:
            message += "Robot:\n no: {}\n".format(rob.robotNo)
            message += " state: {}\n".format(rob.state)        
        return message+"\n"
    
    # -------------------------------------------------------------------------
    def getPosesArray(self):
    # -------------------------------------------------------------------------
        poses = np.zeros((self.nbOfRobots, self.robot[0].stateDim))
        for i in range(self.nbOfRobots):
            poses[i,:] = self.robot[i].state
        
        return poses

# ====================== end of class Fleet ===================================





   
# ============================== MAIN =========================================        
if __name__=='__main__':
# =============================================================================
    
    initState = np.array([-1., 2.])
    robot = Robot(dynamics='singleIntegrator2D', robotNo=0, initState=initState)    
    
    print(robot)
    
    
   
    nbOfRobots = 8   
    
    fleet = Fleet(nbOfRobots, dynamics='singleIntegrator2D')#, initStates=initState)    
    
    for i in range(nbOfRobots):
        fleet.robot[i].state = 20*np.random.rand(2)-10  # random init btw -10, +10
    
    print(fleet) 
    
    print(fleet.getPosesArray())