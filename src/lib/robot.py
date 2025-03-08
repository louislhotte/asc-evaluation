import math
import numpy as np
import matplotlib.pyplot as plt

def si_to_uni(vx, vy, theta, kp=1.0):
    V = np.sqrt(vx**2 + vy**2)
    theta_ref = np.arctan2(vy, vx)
    if math.fabs(theta - theta_ref) > math.pi:
        theta_ref += math.copysign(2 * math.pi, theta)
    omega = kp * (theta_ref - theta)
    return V, omega

class Robot:
    def __init__(self, dynamics='singleIntegrator2D', robotNo=0, initState=[]):
        self.robotNo = robotNo
        self.dynamics = dynamics
        if dynamics == 'singleIntegrator2D':
            self.stateDim = 2
            if len(initState) == 0:
                self.state = np.array([0., 0.])
            else:
                self.state = initState
            self.ctrlDim = 2
            self.ctrl = np.array([0., 0.])
        if dynamics == 'unicycle':
            self.stateDim = 3
            if len(initState) == 0:
                self.state = np.array([0., 0., 0.])
            else:
                self.state = initState
            self.ctrlDim = 2
            self.ctrl = np.array([0., 0.])
    
    def setCtrl(self, ctrl):
        self.ctrl = ctrl      
            
    def integrateMotion(self, Te):
        if self.dynamics == 'singleIntegrator2D':
            self.state = self.state + Te * self.ctrl
        if self.dynamics == 'unicycle':
            self.state[0] = self.state[0] + Te * self.ctrl[0] * np.cos(self.state[2])
            self.state[1] = self.state[1] + Te * self.ctrl[0] * np.sin(self.state[2])
            self.state[2] = self.state[2] + Te * self.ctrl[1]
  
    def __repr__(self):
        message = "\nRobot:\n index: {}\n".format(self.robotNo)
        message += " state: {}".format(self.state)        
        return message + "\n"
    
    def __str__(self):
        message = "\nRobot:\n no: {}\n".format(self.robotNo)
        message += " state: {}".format(self.state)        
        return message + "\n"

class Fleet:
    def __init__(self, nbOfRobots, dynamics='singleIntegrator2D', initStates=[]):
        self.nbOfRobots = nbOfRobots
        self.robot = []
        for robotNo in range(self.nbOfRobots):
            if len(initStates) > 0:
                self.robot.append(Robot(dynamics, robotNo, initStates[robotNo, :]))
            else:
                self.robot.append(Robot(dynamics, robotNo, initStates))
        
    def integrateMotion(self, Te):
        for i in range(self.nbOfRobots):
            self.robot[i].integrateMotion(Te)
    
    def __repr__(self):
        message = "\nFleet\n"
        for rob in self.robot:
            message += " Robot:\n no: {}\n".format(rob.robotNo)
            message += "  state: {}\n".format(rob.state)
        return message + "\n"
    
    def __str__(self):
        message = "\nFleet\n"
        for rob in self.robot:
            message += "Robot:\n no: {}\n".format(rob.robotNo)
            message += " state: {}\n".format(rob.state)
        return message + "\n"
    
    def getPosesArray(self):
        poses = np.zeros((self.nbOfRobots, self.robot[0].stateDim))
        for i in range(self.nbOfRobots):
            poses[i, :] = self.robot[i].state
        return poses

if __name__=='__main__':
    initState = np.array([-1., 2.])
    robot = Robot(dynamics='singleIntegrator2D', robotNo=0, initState=initState)  
    nbOfRobots = 8   
    fleet = Fleet(nbOfRobots, dynamics='singleIntegrator2D')
    
    for i in range(nbOfRobots):
        fleet.robot[i].state = 20 * np.random.rand(2) - 10
