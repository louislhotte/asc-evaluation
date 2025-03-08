# -*- coding: utf-8 -*-
"""
Simulation Class with Potential options

author: S. Bertrand, 2024
"""
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import lib.robot as robot_lib
#import robot as robot_lib
from lib.potential import Potential



# =============================================================================
# adapted from Python Robotics
def plot_robot(x, y, theta, x_traj, y_traj, scale=1, color='k'):  # pragma: no cover
# =============================================================================
    
    if (theta==None):  # point robot
        plt.plot(x, y, marker='o', color=color)

    else: # arrow with orientation
        # Corners of triangular vehicle when pointing to the right (0 radians)
        p1 = np.array([0.5*scale, 0*scale, 1]).T
        p2 = np.array([-0.5*scale, 0.25*scale, 1]).T
        p3 = np.array([-0.5*scale, -0.25*scale, 1]).T
    
        rot_matrix = np.array([
            [np.cos(theta), -np.sin(theta), x],
            [np.sin(theta), np.cos(theta), y],
            [0, 0, 1]
        ])
    
        p1 = np.matmul(rot_matrix, p1)
        p2 = np.matmul(rot_matrix, p2)
        p3 = np.matmul(rot_matrix, p3)
    
        plt.plot([p1[0], p2[0]], [p1[1], p2[1]], color+'-')
        plt.plot([p2[0], p3[0]], [p2[1], p3[1]], color+'-')
        plt.plot([p3[0], p1[0]], [p3[1], p1[1]], color+'-')

    # trajectory
    plt.plot(x_traj, y_traj, color+'-') # trajectory
    # initial position
    plt.plot(x_traj[0], y_traj[0], marker='+', color=color)


# =============================================================================
class RobotSimulation:
# =============================================================================

    # -------------------------------------------------------------------------
    def __init__(self, robot, t0=0.0, tf=10.0, dt=0.01):
    # -------------------------------------------------------------------------

        # associated robot
        self.robot = robot

        # time
        self.t0 = t0 # init time of simulation (in sec)
        self.tf = tf # final time of simulation (in sec)
        self.dt = dt # sampling period for numerical integration (in sec)
        self.t = np.arange(t0, tf-t0+dt, dt) # vector of time stamps

        # to save robot state and input during simulation
        self.state = np.zeros( [ int(self.t.shape[0]), self.robot.stateDim ] )
        self.ctrl = np.zeros( [ int(self.t.shape[0]), self.robot.ctrlDim ] )
        
        
        # index of current stored data (from 0 to len(self.t)-1 )
        self.currentIndex = 0



    # -------------------------------------------------------------------------
    def addDataFromRobot(self, robot):
    # -------------------------------------------------------------------------
        # store state data
        for i in range(0,robot.stateDim):    
            self.state[self.currentIndex,i] = robot.state[i]
        # store ctrl data
        for i in range(0,robot.ctrlDim):    
            self.ctrl[self.currentIndex,i] = robot.ctrl[i]
        # increment storage index
        self.currentIndex = self.currentIndex + 1


    
    # -------------------------------------------------------------------------
    def plotXY(self, figNo = 1, xmin=-10, xmax=10, ymin=-10, ymax=10, steps=None, color='b', potential=None):
    # -------------------------------------------------------------------------
    
        # plot 2D position
        fig1 = plt.figure(figNo)
    
        
        graph = fig1.add_subplot(111, aspect='equal', autoscale_on=False, xlim=(xmin, xmax), ylim=(ymin, ymax))
    
        if (potential!=None):
            potential.plot(noFigure=figNo,fig=fig1,ax=graph, colorbar=True)
    
    
        if (steps==None):
            graph.plot(self.state[:,0], self.state[:,1], color=color)            
        else:
            graph.plot(self.state[::steps,0], self.state[::steps,1], marker='.', color=color)    
        graph.plot(self.state[0,0], self.state[0,1], marker='+', color=color) # initial position
        graph.grid(True)
        graph.set_xlabel('x (m)')
        graph.set_ylabel('y (m)')
        
        
        



    # -------------------------------------------------------------------------
    def plotState(self, figNo = 1, xmin=-10, xmax=10, ymin=-10, ymax=10, steps=None, color='b'):
    # -------------------------------------------------------------------------        

        # plot x Vs time
        fig2 = plt.figure(figNo)
        graph= fig2.add_subplot(111)
        graph.plot(self.t[::steps], self.state[::steps,0], color=color)        
        graph.grid(True)
        graph.set_xlabel('t (s)')
        graph.set_ylabel('x (m)')
        
        # plot y Vs time
        fig3 = plt.figure(figNo+1)
        graph= fig3.add_subplot(111)
        graph.plot(self.t[::steps], self.state[::steps,1], color=color)        
        graph.grid(True)
        graph.set_xlabel('t (s)')
        graph.set_ylabel('y (m)')
        
        # plot theta Vs time (for unicycle dynamics only)
        if (self.robot.dynamics=='unicycle'):
            fig4 = plt.figure(figNo+2)
            graph= fig4.add_subplot(111)
            graph.plot(self.t[::steps], self.state[::steps,2], color=color)        
            graph.grid(True)
            graph.set_xlabel('t (s)')
            graph.set_ylabel('theta (rad)')
            
            
            
    # -------------------------------------------------------------------------
    def plotCtrl(self, figNo = 1, xmin=-10, xmax=10, ymin=-10, ymax=10, steps=None, color='b'):
    # -------------------------------------------------------------------------        

        # plot ux Vs time
        fig2 = plt.figure(figNo)
        graph= fig2.add_subplot(111)
        graph.plot(self.t[::steps], self.ctrl[::steps,0], color=color)        
        graph.grid(True)
        graph.set_xlabel('t (s)')
        if (self.robot.dynamics=='unicycle'):
            graph.set_ylabel('V (m/s)')
        if (self.robot.dynamics=='singleIntegrator2D'):
            graph.set_ylabel('ux (m/s)')
        
        # plot uy Vs time
        fig3 = plt.figure(figNo+1)
        graph= fig3.add_subplot(111)
        graph.plot(self.t[::steps], self.ctrl[::steps,1], color=color)        
        graph.grid(True)
        graph.set_xlabel('t (s)')
        if (self.robot.dynamics=='unicycle'):
            graph.set_ylabel('omega (rad/s)')
        if (self.robot.dynamics=='singleIntegrator2D'):
            graph.set_ylabel('uy (m/s)')
        

    # -------------------------------------------------------------------------       
    def animation(self, figNo=1, xmin=-10, xmax=10, ymin=-10, ymax=10, color='b', robot_scale=0.1, pause=0.0001, potential=None):
    # -------------------------------------------------------------------------
     
        plt.figure(figNo)
        global stop_anim
        stop_anim = False
        
        
        def on_escape(event):
            global stop_anim
            if event.key == 'escape':
                stop_anim = True

        plt.xlabel("x (m)")
        plt.ylabel("y (m)")
        plt.grid(True)
        plt.xlim((xmin, xmax))
        plt.ylim((ymin, ymax))


        i=0
        while (i<len(self.t))&(stop_anim==False):
        #for i in range(len(self.t)):
            t = self.t[i]
            x = self.state[i,0]
            y = self.state[i,1]
            x_traj = self.state[:i+1,0]
            y_traj = self.state[:i+1,1]
            if len(self.state[i])>2:
                theta = self.state[i,2]
            else:
                theta = None
            
            # clear plot
            plt.cla()
            #plt.axis("equal")
    
            if (potential!=None):
                potential.plot(noFigure=figNo,fig=plt.gcf(),ax=plt.gca(), colorbar=False)
        
    
            # robot and trajectory
            plot_robot(x, y, theta, x_traj, y_traj, scale=robot_scale, color=color)

            '''
            plt.xlabel("x (m)")
            plt.ylabel("y (m)")
            plt.grid(True)
            plt.xlim((xmin, xmax))
            plt.ylim((ymin, ymax))
            '''
            plt.title("(press Escape to stop animation)\n" + "Time: " + str(round(t, 2)) + "s / " + str(round(self.t[-1],2)) + "s" )

            
            # for stopping simulation with the esc key.
            plt.gcf().canvas.mpl_connect('key_release_event', on_escape)#lambda event: [return True if event.key == 'escape' else None])
            
            plt.pause(pause)
            
            i=i+1
            
    
            

# ====================== end of class RobotSimulation==========================




# =============================================================================
class FleetSimulation:
# =============================================================================

    # -------------------------------------------------------------------------
    def __init__(self, fleet, t0=0.0, tf=10.0, dt=0.01):
    # -------------------------------------------------------------------------

        self.nbOfRobots = fleet.nbOfRobots
        
        self.robotSimulation = []
        
        for i in range(self.nbOfRobots):
            self.robotSimulation.append( RobotSimulation(fleet.robot[i], t0, tf, dt) )

        # time
        self.t0 = t0 # init time of simulation (in sec)
        self.tf = tf # final time of simulation (in sec)
        self.dt = dt # sampling period for numerical integration (in sec)
        self.t = np.arange(t0, tf-t0+dt, dt) # vector of time stamps


# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# !!!!!!! PRévoir un add data from robot avec l index du robot
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    # -------------------------------------------------------------------------
    def addDataFromFleet(self, fleet):
    # -------------------------------------------------------------------------
        for i in range(self.nbOfRobots):
            self.robotSimulation[i].addDataFromRobot(fleet.robot[i])


    # -----------------------------------------------------------------------------------
    def plotXY(self, figNo=1,  xmin=-10, xmax=10, ymin=-10, ymax=10, steps=None, links=False, potential=None):
    # -----------------------------------------------------------------------------------

        fig1 = plt.figure(figNo)
        graph = fig1.add_subplot(111, aspect='equal', autoscale_on=False, xlim=(xmin, xmax), ylim=(ymin, ymax))

        if (potential!=None):
            potential.plot(noFigure=figNo,fig=fig1,ax=graph, colorbar=True)



        colorList = ['r', 'g', 'b', 'y', 'c', 'm', 'k']
        for i in range(self.nbOfRobots):

            i_color = np.mod(i, len(colorList))  # use last color of the list if less colors than nb of robots
            if (steps==None):
                graph.plot(self.robotSimulation[i].state[:,0], self.robotSimulation[i].state[:,1], color=colorList[i_color])            
            else:
                graph.plot(self.robotSimulation[i].state[::steps,0], self.robotSimulation[i].state[::steps,1], marker='.', color=colorList[i_color])      
            graph.plot(self.robotSimulation[i].state[0,0], self.robotSimulation[i].state[0,1], marker='+', color=colorList[i_color])    # initial position
        
        
        if (links==True):
            plt.gca().set_prop_cycle(None)        
        
            for tt in range(0, int(self.t.shape[0]))[::steps]:
                for i in range(0,self.nbOfRobots):
                    i_color = np.mod(i, len(colorList))  # use last color of the list if less colors than nb of robots
                    for j in range(0, self.nbOfRobots):
                        xi = self.robotSimulation[i].state[tt,0]
                        yi = self.robotSimulation[i].state[tt,1]
                        xj = self.robotSimulation[j].state[tt,0]
                        yj = self.robotSimulation[j].state[tt,1]
                        graph.plot([xi, xj], [yi, yj], color='grey', alpha = 0.3, linestyle='--')            
                        graph.plot(xi, yi, marker = '8', linestyle="None", markersize=5, color=colorList[i_color] )
            
            plt.gca().set_prop_cycle(None)
            
        
        graph.grid(True)
        graph.set_xlabel('x (m)')
        graph.set_ylabel('y (m)')
        
        
    # -----------------------------------------------------------------------------------
    def plotState(self, figNo=1,  xmin=-10, xmax=10, ymin=-10, ymax=10):
    # -----------------------------------------------------------------------------------
        
        colorList = ['r', 'g', 'b', 'y', 'c', 'm', 'k']
        
        # plot x Vs time
        fig2 = plt.figure(figNo)
        graph= fig2.add_subplot(111)
        for i in range(0,self.nbOfRobots):
            i_color = np.mod(i, len(colorList))  # use last color of the list if less colors than nb of robots
            graph.plot(self.t, self.robotSimulation[i].state[:,0], color=colorList[i_color])
        graph.grid(True)
        graph.set_xlabel('t (s)')
        graph.set_ylabel('x (m)')
        
        # plot y Vs time
        fig3 = plt.figure(figNo+1)
        graph= fig3.add_subplot(111)
        for i in range(0,self.nbOfRobots):
            i_color = np.mod(i, len(colorList))  # use last color of the list if less colors than nb of robots
            graph.plot(self.t, self.robotSimulation[i].state[:,1], color=colorList[i_color])
        graph.grid(True)
        graph.set_xlabel('t (s)')
        graph.set_ylabel('y (m)')
        
        # plot theta Vs time (for unicycle dynamics only)
        if (self.robotSimulation[0].robot.dynamics=='unicycle'):
            fig4 = plt.figure(figNo+2)
            graph= fig4.add_subplot(111)
            for i in range(0,self.nbOfRobots):
                i_color = np.mod(i, len(colorList))  # use last color of the list if less colors than nb of robots
                graph.plot(self.t, self.robotSimulation[i].state[:,2], color=colorList[i_color])
            graph.grid(True)
            graph.set_xlabel('t (s)')
            graph.set_ylabel('theta (rad)')
            
            
            
            
    # -----------------------------------------------------------------------------------
    def plotCtrl(self, figNo=1,  xmin=-10, xmax=10, ymin=-10, ymax=10):
    # -----------------------------------------------------------------------------------
        
        colorList = ['r', 'g', 'b', 'y', 'c', 'm', 'k']
        
        # plot x Vs time
        fig2 = plt.figure(figNo)
        graph= fig2.add_subplot(111)
        for i in range(0,self.nbOfRobots):
            i_color = np.mod(i, len(colorList))  # use last color of the list if less colors than nb of robots
            graph.plot(self.t, self.robotSimulation[i].ctrl[:,0], color=colorList[i_color])
        graph.grid(True)
        graph.set_xlabel('t (s)')
        if (self.robotSimulation[0].robot.dynamics=='unicycle'):
            graph.set_ylabel('V (m/s)')
        if (self.robotSimulation[0].robot.dynamics=='singleIntegrator2D'):
            graph.set_ylabel('ux (m/s)')
        
        # plot y Vs time
        fig3 = plt.figure(figNo+1)
        graph= fig3.add_subplot(111)
        for i in range(0,self.nbOfRobots):
            i_color = np.mod(i, len(colorList))  # use last color of the list if less colors than nb of robots
            graph.plot(self.t, self.robotSimulation[i].ctrl[:,1], color=colorList[i_color])
        graph.grid(True)
        graph.set_xlabel('t (s)')
        if (self.robotSimulation[0].robot.dynamics=='unicycle'):
            graph.set_ylabel('omega (rad/s)')
        if (self.robotSimulation[0].robot.dynamics=='singleIntegrator2D'):
            graph.set_ylabel('uy (m/s)')
        
    
    # -------------------------------------------------------------------------       
    def animation(self, figNo=1, xmin=-10, xmax=10, ymin=-10, ymax=10, robot_scale=0.1, pause=0.0001, potential=None):
    # -------------------------------------------------------------------------
     
        plt.figure(figNo)
        global stop_anim
        stop_anim = False
        
        
        def on_escape(event):
            global stop_anim
            if event.key == 'escape':
                stop_anim = True


        colorList = ['r', 'g', 'b', 'y', 'c', 'm', 'k']
      
        i=0
        while (i<len(self.t))&(stop_anim==False):
            # clear plot
            plt.cla()
            #plt.axis("equal")
            
            if (potential!=None):
                potential.plot(noFigure=figNo,fig=plt.gcf(),ax=plt.gca(), colorbar=False)
            
            
            for i_rob in range(self.nbOfRobots):
                
                i_color = np.mod(i_rob, len(colorList))  # use last color of the list if less colors than nb of robots            

    
                t = self.t[i]
                x = self.robotSimulation[i_rob].state[i,0]
                y = self.robotSimulation[i_rob].state[i,1]
                x_traj = self.robotSimulation[i_rob].state[:i+1,0]
                y_traj = self.robotSimulation[i_rob].state[:i+1,1]
                if len(self.robotSimulation[i_rob].state[i])>2:
                    theta = self.robotSimulation[i_rob].state[i,2]
                else:
                    theta = None
                                
                # robot and trajectory
                plot_robot(x, y, theta, x_traj, y_traj, scale=robot_scale, color=colorList[i_color])


            plt.xlabel("x (m)")
            plt.ylabel("y (m)")
            plt.grid(True)
            plt.xlim((xmin, xmax))
            plt.ylim((ymin, ymax))
            plt.title("(press Escape to stop animation)\n" + "Time: " + str(round(t, 2)) + "s / " + str(round(self.t[-1],2)) + "s" )
            
            # for stopping simulation with the esc key.
            plt.gcf().canvas.mpl_connect('key_release_event', on_escape)#lambda event: [return True if event.key == 'escape' else None])
            
            plt.pause(pause)
            
            i=i+1
    '''
    # -----------------------------------------------------------------------------------------
    def plotFleet(self, figNo = 1, xmin=-10, xmax=10, ymin=-10, ymax=10, mod=None, links=True):
    # ------------------------------------------------------------------------------------------
        
        fig1 = plt.figure(figNo)
        graph = fig1.add_subplot(111, aspect='equal', autoscale_on=False, xlim=(xmin, xmax), ylim=(ymin, ymax))
#            graph.plot(self.state[0,:], self.state[1,:], color = 'r')
#            graph.plot(self.state[0,-1], self.state[1,-1], color = 'r', marker='o')
        
        #print int(self.t.shape[0])

        plt.gca().set_prop_cycle(None)        
        
        for tt in range(0, int(self.t.shape[0]))[::mod]:
            for i in range(0,self.nbOfRobots):
                for j in range(0, self.nbOfRobots):
                    xi = self.robotSimulation[i].state[tt,0]
                    yi = self.robotSimulation[i].state[tt,1]
                    xj = self.robotSimulation[j].state[tt,0]
                    yj = self.robotSimulation[j].state[tt,1]
                    if (links==True):
                        graph.plot([xi, xj], [yi, yj], color='grey', alpha = 0.3, linestyle='--')            
                graph.plot(xi, yi, marker = '8', linestyle="None", markersize=5 )
            plt.gca().set_prop_cycle(None)
        graph.grid(True)
        graph.set_xlabel('x (m)')
        graph.set_ylabel('y (m)')
      '''
        
      # -----------------------------------------------------------------------------------------
      #def animate(self, figNo = 1, xmin=-10, xmax=10, ymin=-10, ymax=10, mod=None, links=True):
    
    
   
# ====================== end of class RobotSimulation==========================



   
# ============================== MAIN =========================================        
if __name__=='__main__':
# =============================================================================
    
    test_no = 1


    ## ---- one robot simulation test  (Single Integrator dynamics)
    if (test_no==1):
        initState = np.array([0., 0.])
        robot = robot_lib.Robot(dynamics='singleIntegrator2D', robotNo=0, initState=initState)    
        
        Te = 0.01
        simulation = RobotSimulation(robot, t0=0.0, tf=20.0, dt=Te)
        
        # reference
        referenceState= np.array([5.,5.])
       
        # control gain
        kp = 0.4
        
        for t in simulation.t:
            
            robot.ctrl = kp* (referenceState - robot.state)
            
            simulation.addDataFromRobot(robot)
        
            robot.integrateMotion(Te)
    
        print(robot)
        
        simulation.animation(figNo=1, pause=0.00001, color='r', robot_scale=1.0) 
        simulation.plotXY(figNo=2, color='r')
        simulation.plotState(figNo=3, color='r')
        simulation.plotCtrl(figNo=5, color='r')






    ## ---- one robot simulation test (Unicycle dynamics)
    if (test_no==2):
        initState = np.array([0., 0., 0.])
        robot = robot_lib.Robot(dynamics='unicycle', robotNo=0, initState=initState)    
        
        Te = 0.01
        simulation = RobotSimulation(robot, t0=0.0, tf=20.0, dt=Te)
        
        # reference
        refPosition= np.array([5.,5.])
       
        # control gain
        kp = 0.4
        
        for t in simulation.t:
            
            deltaX = refPosition[0]-robot.state[0]
            deltaY = refPosition[1]-robot.state[1]
            V = kp*np.sqrt( deltaX**2 + deltaY**2)
            
            theta_ref = np.arctan2(refPosition[1]-robot.state[1],refPosition[0]-robot.state[0])
            # avoid U-turns    
            if math.fabs(robot.state[2]-theta_ref)>math.pi:
                    theta_ref += math.copysign(2*math.pi, robot.state[2])
            
            omega = 4*kp*(theta_ref-robot.state[2])
            
            robot.ctrl[0] = V
            robot.ctrl[1] = omega
            
            simulation.addDataFromRobot(robot)
        
            robot.integrateMotion(Te)
    
        print(robot)
    
        simulation.animation(figNo=1, pause=0.00001, color='r', robot_scale=1.0)    
        simulation.plotXY(figNo=2, color='r')
        simulation.plotState(figNo=3, color='r')
        simulation.plotCtrl(figNo=6, color='r')

   

    # ---- fleet simulation test (single integrator dynamics)
    if (test_no==3):
        nbOfRobots = 8  
        
        fleet = robot_lib.Fleet(nbOfRobots, dynamics='singleIntegrator2D')#, initState=initState)    
        
        for i in range(0, nbOfRobots):
            fleet.robot[i].state = 20*np.random.rand(2)-10  # random init btw -10, +10
        
        Te = 0.01
        simulation = FleetSimulation(fleet, t0=0.0, tf=20.0, dt=Te)
        
       
        # control gain
        kp = 0.4
    
        
        for t in simulation.t:
    
            #proportional control law to common reference state 
            #referenceState= np.array([2., 1.])
            #for r in range(0, fleet.nbOfRobots):
            #    fleet.robot[r].ctrl = kp* (referenceState - fleet.robot[r].state)
     
            # consensus
            
    
            for r in range(fleet.nbOfRobots):
                fleet.robot[r].ctrl = np.zeros(2)
                for n in range(fleet.nbOfRobots):
                    fleet.robot[r].ctrl += kp* (fleet.robot[n].state - fleet.robot[r].state) / fleet.nbOfRobots
            
    
            simulation.addDataFromFleet(fleet)
        
            fleet.integrateMotion(Te)
    
    
        simulation.animation(figNo=1, pause=0.00001, robot_scale=1.0)   
        
        simulation.plotXY(figNo=2)
        simulation.plotState(figNo=3)
        simulation.plotCtrl(figNo=5)
        #simulation.plotXY(figNo=10, steps=100, links=True)





    # ---- fleet simulation test (unicycle dynamics)
    if (test_no==4):    
        nbOfRobots = 4  
        
        fleet = robot_lib.Fleet(nbOfRobots, dynamics='unicycle')#, initState=initState)    
        
        for i in range(0, nbOfRobots):
            fleet.robot[i].state[0] = 20*np.random.rand(1)-10  # random init btw -10, +10
            fleet.robot[i].state[1] = 20*np.random.rand(1)-10  # random init btw -10, +10
            fleet.robot[i].state[2] = 2*np.pi*np.random.rand(1)-np.pi  # random init btw -pi, pi
        
        Te = 0.01
        simulation = FleetSimulation(fleet, t0=0.0, tf=20.0, dt=Te)
        
        
        # control gain
        kp = 0.4
        
        
        for t in simulation.t:
        
            #proportional control law to common reference state 
            #referenceState= np.array([2., 1.])
            #for r in range(0, fleet.nbOfRobots):
            #    fleet.robot[r].ctrl = kp* (referenceState - fleet.robot[r].state)
        
            # consensus
            
        
            for r in range(fleet.nbOfRobots):
            
                u_cart = np.zeros(2)
                
                for n in range(fleet.nbOfRobots):
                    u_cart += kp* (fleet.robot[n].state[0:2] - fleet.robot[r].state[0:2]) / fleet.nbOfRobots
            
                V = np.sqrt( u_cart[0]**2 + u_cart[1]**2)
                
                theta_ref = np.arctan2(u_cart[1], u_cart[0])
                # avoid U-turns    
                if math.fabs(fleet.robot[r].state[2]-theta_ref)>math.pi:
                        theta_ref += math.copysign(2*math.pi, fleet.robot[r].state[2])
                
                omega = 4*kp*(theta_ref-fleet.robot[r].state[2])
                
            
                fleet.robot[r].ctrl = np.zeros(2)
                fleet.robot[r].ctrl[0] = V
                fleet.robot[r].ctrl[1] = omega
            
            
        
            simulation.addDataFromFleet(fleet)
        
            fleet.integrateMotion(Te)
        
        
        simulation.animation(figNo=1, pause=0.00001, robot_scale=1.0)   
        
        simulation.plotXY(figNo=2)
        simulation.plotXY(figNo=3, steps=100, links=True)
        simulation.plotState(figNo=4)
        simulation.plotCtrl(figNo=7)
        
    
