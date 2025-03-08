"""
Simulation Class with Potential options
Author: S. Bertrand, 2024
"""

import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import robot as robot_lib
from potential import Potential


def plot_robot(x, y, theta, x_traj, y_traj, scale=1, color='k'):
    if theta is None:  # point robot
        plt.plot(x, y, marker='o', color=color)
    else:  # robot with orientation (arrow)
        # Define triangular vehicle shape when pointing right (0 radians)
        p1 = np.array([0.5 * scale, 0 * scale, 1]).T
        p2 = np.array([-0.5 * scale, 0.25 * scale, 1]).T
        p3 = np.array([-0.5 * scale, -0.25 * scale, 1]).T

        rot_matrix = np.array([
            [np.cos(theta), -np.sin(theta), x],
            [np.sin(theta), np.cos(theta), y],
            [0, 0, 1]
        ])

        p1 = np.matmul(rot_matrix, p1)
        p2 = np.matmul(rot_matrix, p2)
        p3 = np.matmul(rot_matrix, p3)

        plt.plot([p1[0], p2[0]], [p1[1], p2[1]], color + '-')
        plt.plot([p2[0], p3[0]], [p2[1], p3[1]], color + '-')
        plt.plot([p3[0], p1[0]], [p3[1], p1[1]], color + '-')

    plt.plot(x_traj, y_traj, color + '-')
    plt.plot(x_traj[0], y_traj[0], marker='+', color=color)


class RobotSimulation:
    def __init__(self, robot, t0=0.0, tf=10.0, dt=0.01):
        self.robot = robot
        self.t0 = t0
        self.tf = tf
        self.dt = dt
        self.t = np.arange(t0, tf - t0 + dt, dt)
        self.state = np.zeros([len(self.t), self.robot.stateDim])
        self.ctrl = np.zeros([len(self.t), self.robot.ctrlDim])
        self.currentIndex = 0

    def addDataFromRobot(self, robot):
        for i in range(robot.stateDim):
            self.state[self.currentIndex, i] = robot.state[i]
        for i in range(robot.ctrlDim):
            self.ctrl[self.currentIndex, i] = robot.ctrl[i]
        self.currentIndex += 1

    def plotXY(self, figNo=1, xmin=-10, xmax=10, ymin=-10, ymax=10, steps=None, color='b', potential=None):
        fig = plt.figure(figNo)
        ax = fig.add_subplot(111, aspect='equal', autoscale_on=False, xlim=(xmin, xmax), ylim=(ymin, ymax))
        if potential is not None:
            potential.plot(noFigure=figNo, fig=fig, ax=ax, colorbar=True)
        if steps is None:
            ax.plot(self.state[:, 0], self.state[:, 1], color=color)
        else:
            ax.plot(self.state[::steps, 0], self.state[::steps, 1], marker='.', color=color)
        ax.plot(self.state[0, 0], self.state[0, 1], marker='+', color=color)
        ax.grid(True)
        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')

    def plotState(self, figNo=1, xmin=-10, xmax=10, ymin=-10, ymax=10, steps=None, color='b'):
        fig1 = plt.figure(figNo)
        ax1 = fig1.add_subplot(111)
        ax1.plot(self.t[::steps], self.state[::steps, 0], color=color)
        ax1.grid(True)
        ax1.set_xlabel('t (s)')
        ax1.set_ylabel('x (m)')

        fig2 = plt.figure(figNo + 1)
        ax2 = fig2.add_subplot(111)
        ax2.plot(self.t[::steps], self.state[::steps, 1], color=color)
        ax2.grid(True)
        ax2.set_xlabel('t (s)')
        ax2.set_ylabel('y (m)')

        if self.robot.dynamics == 'unicycle':
            fig3 = plt.figure(figNo + 2)
            ax3 = fig3.add_subplot(111)
            ax3.plot(self.t[::steps], self.state[::steps, 2], color=color)
            ax3.grid(True)
            ax3.set_xlabel('t (s)')
            ax3.set_ylabel('theta (rad)')

    def plotCtrl(self, figNo=1, xmin=-10, xmax=10, ymin=-10, ymax=10, steps=None, color='b'):
        fig1 = plt.figure(figNo)
        ax1 = fig1.add_subplot(111)
        ax1.plot(self.t[::steps], self.ctrl[::steps, 0], color=color)
        ax1.grid(True)
        ax1.set_xlabel('t (s)')
        if self.robot.dynamics == 'unicycle':
            ax1.set_ylabel('V (m/s)')
        elif self.robot.dynamics == 'singleIntegrator2D':
            ax1.set_ylabel('ux (m/s)')

        fig2 = plt.figure(figNo + 1)
        ax2 = fig2.add_subplot(111)
        ax2.plot(self.t[::steps], self.ctrl[::steps, 1], color=color)
        ax2.grid(True)
        ax2.set_xlabel('t (s)')
        if self.robot.dynamics == 'unicycle':
            ax2.set_ylabel('omega (rad/s)')
        elif self.robot.dynamics == 'singleIntegrator2D':
            ax2.set_ylabel('uy (m/s)')

    def animation(self, figNo=1, xmin=-10, xmax=10, ymin=-10, ymax=10,
                  color='b', robot_scale=0.1, pause=0.0001, potential=None):
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

        i = 0
        while i < len(self.t) and not stop_anim:
            t = self.t[i]
            x = self.state[i, 0]
            y = self.state[i, 1]
            x_traj = self.state[:i + 1, 0]
            y_traj = self.state[:i + 1, 1]
            theta = self.state[i, 2] if len(self.state[i]) > 2 else None

            plt.cla()
            if potential is not None:
                potential.plot(noFigure=figNo, fig=plt.gcf(), ax=plt.gca(), colorbar=False)

            plot_robot(x, y, theta, x_traj, y_traj, scale=robot_scale, color=color)
            plt.title("(press Escape to stop animation)\nTime: " +
                      str(round(t, 2)) + "s / " + str(round(self.t[-1], 2)) + "s")
            plt.gcf().canvas.mpl_connect('key_release_event', on_escape)
            plt.pause(pause)
            i += 1


class FleetSimulation:
    def __init__(self, fleet, t0=0.0, tf=10.0, dt=0.01):
        self.nbOfRobots = fleet.nbOfRobots
        self.robotSimulation = [RobotSimulation(fleet.robot[i], t0, tf, dt)
                                for i in range(self.nbOfRobots)]
        self.t0 = t0
        self.tf = tf
        self.dt = dt
        self.t = np.arange(t0, tf - t0 + dt, dt)

    def addDataFromFleet(self, fleet):
        for i in range(self.nbOfRobots):
            self.robotSimulation[i].addDataFromRobot(fleet.robot[i])

    def plotXY(self, figNo=1, xmin=-10, xmax=10, ymin=-10, ymax=10, steps=None,
               links=False, potential=None):
        fig = plt.figure(figNo)
        ax = fig.add_subplot(111, aspect='equal', autoscale_on=False, xlim=(xmin, xmax), ylim=(ymin, ymax))
        if potential is not None:
            potential.plot(noFigure=figNo, fig=fig, ax=ax, colorbar=True)

        colorList = ['r', 'g', 'b', 'y', 'c', 'm', 'k']
        for i in range(self.nbOfRobots):
            c = colorList[i % len(colorList)]
            if steps is None:
                ax.plot(self.robotSimulation[i].state[:, 0], self.robotSimulation[i].state[:, 1], color=c)
            else:
                ax.plot(self.robotSimulation[i].state[::steps, 0],
                        self.robotSimulation[i].state[::steps, 1], marker='.', color=c)
            ax.plot(self.robotSimulation[i].state[0, 0], self.robotSimulation[i].state[0, 1],
                    marker='+', color=c)

        if links:
            ax.set_prop_cycle(None)
            for tt in range(0, len(self.t), steps or 1):
                for i in range(self.nbOfRobots):
                    c = colorList[i % len(colorList)]
                    for j in range(self.nbOfRobots):
                        xi = self.robotSimulation[i].state[tt, 0]
                        yi = self.robotSimulation[i].state[tt, 1]
                        xj = self.robotSimulation[j].state[tt, 0]
                        yj = self.robotSimulation[j].state[tt, 1]
                        ax.plot([xi, xj], [yi, yj], color='grey', alpha=0.3, linestyle='--')
                    ax.plot(xi, yi, marker='8', linestyle="None", markersize=5, color=c)
            ax.set_prop_cycle(None)

        ax.grid(True)
        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')

    def plotState(self, figNo=1, xmin=-10, xmax=10, ymin=-10, ymax=10):
        colorList = ['r', 'g', 'b', 'y', 'c', 'm', 'k']

        fig1 = plt.figure(figNo)
        ax1 = fig1.add_subplot(111)
        for i in range(self.nbOfRobots):
            c = colorList[i % len(colorList)]
            ax1.plot(self.t, self.robotSimulation[i].state[:, 0], color=c)
        ax1.grid(True)
        ax1.set_xlabel('t (s)')
        ax1.set_ylabel('x (m)')

        fig2 = plt.figure(figNo + 1)
        ax2 = fig2.add_subplot(111)
        for i in range(self.nbOfRobots):
            c = colorList[i % len(colorList)]
            ax2.plot(self.t, self.robotSimulation[i].state[:, 1], color=c)
        ax2.grid(True)
        ax2.set_xlabel('t (s)')
        ax2.set_ylabel('y (m)')

        if self.robotSimulation[0].robot.dynamics == 'unicycle':
            fig3 = plt.figure(figNo + 2)
            ax3 = fig3.add_subplot(111)
            for i in range(self.nbOfRobots):
                c = colorList[i % len(colorList)]
                ax3.plot(self.t, self.robotSimulation[i].state[:, 2], color=c)
            ax3.grid(True)
            ax3.set_xlabel('t (s)')
            ax3.set_ylabel('theta (rad)')

    def plotCtrl(self, figNo=1, xmin=-10, xmax=10, ymin=-10, ymax=10):
        colorList = ['r', 'g', 'b', 'y', 'c', 'm', 'k']

        fig1 = plt.figure(figNo)
        ax1 = fig1.add_subplot(111)
        for i in range(self.nbOfRobots):
            c = colorList[i % len(colorList)]
            ax1.plot(self.t, self.robotSimulation[i].ctrl[:, 0], color=c)
        ax1.grid(True)
        ax1.set_xlabel('t (s)')
        if self.robotSimulation[0].robot.dynamics == 'unicycle':
            ax1.set_ylabel('V (m/s)')
        elif self.robotSimulation[0].robot.dynamics == 'singleIntegrator2D':
            ax1.set_ylabel('ux (m/s)')

        fig2 = plt.figure(figNo + 1)
        ax2 = fig2.add_subplot(111)
        for i in range(self.nbOfRobots):
            c = colorList[i % len(colorList)]
            ax2.plot(self.t, self.robotSimulation[i].ctrl[:, 1], color=c)
        ax2.grid(True)
        ax2.set_xlabel('t (s)')
        if self.robotSimulation[0].robot.dynamics == 'unicycle':
            ax2.set_ylabel('omega (rad/s)')
        elif self.robotSimulation[0].robot.dynamics == 'singleIntegrator2D':
            ax2.set_ylabel('uy (m/s)')

    def animation(self, figNo=1, xmin=-10, xmax=10, ymin=-10, ymax=10,
                  robot_scale=0.1, pause=0.0001, potential=None):
        plt.figure(figNo)
        global stop_anim
        stop_anim = False

        def on_escape(event):
            global stop_anim
            if event.key == 'escape':
                stop_anim = True

        colorList = ['r', 'g', 'b', 'y', 'c', 'm', 'k']
        i = 0
        while i < len(self.t) and not stop_anim:
            plt.cla()
            if potential is not None:
                potential.plot(noFigure=figNo, fig=plt.gcf(), ax=plt.gca(), colorbar=False)

            for i_rob in range(self.nbOfRobots):
                c = colorList[i_rob % len(colorList)]
                t = self.t[i]
                x = self.robotSimulation[i_rob].state[i, 0]
                y = self.robotSimulation[i_rob].state[i, 1]
                x_traj = self.robotSimulation[i_rob].state[:i + 1, 0]
                y_traj = self.robotSimulation[i_rob].state[:i + 1, 1]
                theta = self.robotSimulation[i_rob].state[i, 2] if len(self.robotSimulation[i_rob].state[i]) > 2 else None
                plot_robot(x, y, theta, x_traj, y_traj, scale=robot_scale, color=c)

            plt.xlabel("x (m)")
            plt.ylabel("y (m)")
            plt.grid(True)
            plt.xlim((xmin, xmax))
            plt.ylim((ymin, ymax))
            plt.title("(press Escape to stop animation)\nTime: " +
                      str(round(t, 2)) + "s / " + str(round(self.t[-1], 2)) + "s")
            plt.gcf().canvas.mpl_connect('key_release_event', on_escape)
            plt.pause(pause)
            i += 1


if __name__ == '__main__':
    test_no = 3

    if test_no == 1:
        initState = np.array([0., 0.])
        robot = robot_lib.Robot(dynamics='singleIntegrator2D', robotNo=0, initState=initState)
        Te = 0.01
        simulation = RobotSimulation(robot, t0=0.0, tf=20.0, dt=Te)
        referenceState = np.array([5., 5.])
        kp = 0.4

        for t in simulation.t:
            robot.ctrl = kp * (referenceState - robot.state)
            simulation.addDataFromRobot(robot)
            robot.integrateMotion(Te)

        simulation.animation(figNo=1, pause=0.00001, color='r', robot_scale=1.0)
        simulation.plotXY(figNo=2, color='r')
        simulation.plotState(figNo=3, color='r')
        simulation.plotCtrl(figNo=5, color='r')

    if test_no == 2:
        initState = np.array([0., 0., 0.])
        robot = robot_lib.Robot(dynamics='unicycle', robotNo=0, initState=initState)
        Te = 0.01
        simulation = RobotSimulation(robot, t0=0.0, tf=20.0, dt=Te)
        refPosition = np.array([5., 5.])
        kp = 0.4

        for t in simulation.t:
            deltaX = refPosition[0] - robot.state[0]
            deltaY = refPosition[1] - robot.state[1]
            V = kp * np.sqrt(deltaX**2 + deltaY**2)
            theta_ref = np.arctan2(refPosition[1] - robot.state[1],
                                   refPosition[0] - robot.state[0])
            if abs(robot.state[2] - theta_ref) > math.pi:
                theta_ref += math.copysign(2 * math.pi, robot.state[2])
            omega = 4 * kp * (theta_ref - robot.state[2])
            robot.ctrl[0] = V
            robot.ctrl[1] = omega
            simulation.addDataFromRobot(robot)
            robot.integrateMotion(Te)

        simulation.animation(figNo=1, pause=0.00001, color='r', robot_scale=1.0)
        simulation.plotXY(figNo=2, color='r')
        simulation.plotState(figNo=3, color='r')
        simulation.plotCtrl(figNo=6, color='r')

    if test_no == 3:
        nbOfRobots = 8
        fleet = robot_lib.Fleet(nbOfRobots, dynamics='singleIntegrator2D')
        for i in range(nbOfRobots):
            fleet.robot[i].state = 20 * np.random.rand(2) - 10
        Te = 0.01
        simulation = FleetSimulation(fleet, t0=0.0, tf=20.0, dt=Te)
        kp = 0.4

        for t in simulation.t:
            for r in range(fleet.nbOfRobots):
                fleet.robot[r].ctrl = np.zeros(2)
                for n in range(fleet.nbOfRobots):
                    fleet.robot[r].ctrl += kp * (fleet.robot[n].state - fleet.robot[r].state) / fleet.nbOfRobots
            simulation.addDataFromFleet(fleet)
            fleet.integrateMotion(Te)

        simulation.animation(figNo=1, pause=0.00001, robot_scale=1.0)
        simulation.plotXY(figNo=2)
        simulation.plotState(figNo=3)
        simulation.plotCtrl(figNo=5)

    if test_no == 4:
        nbOfRobots = 4
        fleet = robot_lib.Fleet(nbOfRobots, dynamics='unicycle')
        for i in range(nbOfRobots):
            fleet.robot[i].state[0] = 20 * np.random.rand(1) - 10
            fleet.robot[i].state[1] = 20 * np.random.rand(1) - 10
            fleet.robot[i].state[2] = 2 * np.pi * np.random.rand(1) - np.pi
        Te = 0.01
        simulation = FleetSimulation(fleet, t0=0.0, tf=20.0, dt=Te)
        kp = 0.4

        for t in simulation.t:
            for r in range(fleet.nbOfRobots):
                u_cart = np.zeros(2)
                for n in range(fleet.nbOfRobots):
                    u_cart += kp * (fleet.robot[n].state[0:2] - fleet.robot[r].state[0:2]) / fleet.nbOfRobots
                V = np.sqrt(u_cart[0]**2 + u_cart[1]**2)
                theta_ref = np.arctan2(u_cart[1], u_cart[0])
                if abs(fleet.robot[r].state[2] - theta_ref) > math.pi:
                    theta_ref += math.copysign(2 * math.pi, fleet.robot[r].state[2])
                omega = 4 * kp * (theta_ref - fleet.robot[r].state[2])
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
