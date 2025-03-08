import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

class Potential:
    
    def __init__(self, difficulty=1, random=False):  
        if difficulty < 1 or difficulty > 3:
            raise ValueError("Difficulty must be between 1 and 3.")
        
        self.difficulty = difficulty
        self.random = random
        
        self.xmin, self.xmax, self.xstep = -25., 25., 0.05
        self.ymin, self.ymax, self.ystep = -25., 25., 0.05

        if random:
            xwidth = np.abs(self.xmax - self.xmin)
            ywidth = np.abs(self.ymax - self.ymin)
            self.mu1 = [0.6 * (xwidth * np.random.rand() - xwidth / 2.), 
                        0.6 * (ywidth * np.random.rand() - ywidth / 2.)]
            self.mu2 = [xwidth * np.random.rand() - xwidth / 2., 
                        ywidth * np.random.rand() - ywidth / 2.]
            self.mu3 = [xwidth * np.random.rand() - xwidth / 2., 
                        ywidth * np.random.rand() - ywidth / 2.]
        else:
            self.mu1, self.mu2, self.mu3 = [6, 4], [-2, -2], [-7, 10]

        self.gaussian1 = multivariate_normal(self.mu1, [[1.0, 0.], [0., 1.]])
        self.gaussian2 = multivariate_normal(self.mu2, [[0.5, 0.3], [0.3, 0.5]])
        self.gaussian3 = multivariate_normal(self.mu3, [[0.8, 0.], [0., 0.8]])

        self.weight1, self.weight2, self.weight3 = 10000, 1, 1E-8

        self.mu = [self.mu1]
        self.distribution = [self.gaussian1]
        self.weight = [self.weight1]

        if difficulty > 1:
            self.mu.append(self.mu2)
            self.distribution.append(self.gaussian2)
            self.weight.append(self.weight2)
        if difficulty > 2:
            self.mu.append(self.mu3)
            self.distribution.append(self.gaussian3)
            self.weight.append(self.weight3)


    def value(self, pos):
        """Returns the potential field value at a given position."""
        sumval = sum(self.weight[i] * self.distribution[i].pdf(pos) for i in range(self.difficulty))
        return np.fmax(310. + np.log10(sumval), -10.)

    def grad(self, pos, epsilon=1):
        """Computes the numerical gradient at a given position using finite differences."""
        pos_x = np.array([pos[0] + epsilon, pos[1]])
        pos_y = np.array([pos[0], pos[1] + epsilon])

        grad_x = (self.value(pos_x) - self.value(pos)) / epsilon
        grad_y = (self.value(pos_y) - self.value(pos)) / epsilon

        return np.array([grad_x, grad_y])

    def meanGrad(self, point, epsilon=1.0):
        """Computes mean gradient using neighborhood points."""
        neighbors = [point + np.array(offset) for offset in 
                     [[-epsilon, epsilon], [epsilon, epsilon], [epsilon, -epsilon], [-epsilon, -epsilon]]]
        
        mean_grad = sum(self.grad(pt, epsilon=epsilon) for pt in neighbors) / len(neighbors)
        return -mean_grad  # Negative gradient for descent

    def plot(self, noFigure=None, fig=None, ax=None, colorbar=True):
        """Plots the potential field."""
        x, y = np.mgrid[self.xmin:self.xmax:self.xstep, self.ymin:self.ymax:self.ystep]
        pos = np.dstack((x, y))
        potentialFieldForPlot = self.value(pos)

        if fig is None:
            fig = plt.figure(noFigure if noFigure else 1)
        if ax is None:
            ax = fig.add_subplot(111)
        
        cs = ax.contourf(x, y, potentialFieldForPlot, 20, cmap='BrBG')
        if colorbar:
            fig.colorbar(cs)

        return fig, ax

    def plotQuiverMeanGrad(self, point, epsilon, ax):
        """Plots the gradient as an arrow at a given position."""
        meanGrad = self.meanGrad(point, epsilon)
        ax.quiver(point[0], point[1], meanGrad[0], meanGrad[1])
        ax.text(point[0] + epsilon, point[1], f'{np.arctan2(meanGrad[1], meanGrad[0]) * 180. / np.pi:.2f}')

if __name__ == '__main__':
    plt.close()
    pot = Potential(difficulty=3, random=True)
    fig2, ax2 = pot.plot(2)
    
    for xx in np.arange(-20, 25, 5):
        for yy in np.arange(-20, 25, 5):
            pot.plotQuiverMeanGrad(np.array([xx, yy]), 1.0, ax2)

    plt.show()
