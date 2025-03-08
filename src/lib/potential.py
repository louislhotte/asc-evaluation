# -*- coding: utf-8 -*-
"""
Potential class

(c) S. Bertrand, 2024
"""


import numpy as np

import matplotlib.pyplot as plt

from scipy.stats import multivariate_normal



# =============================================================================
class Potential:
# =============================================================================
    
    # -------------------------------------------------------------------------
    def __init__(self, difficulty=1, random=False):
    # -------------------------------------------------------------------------    
        if (difficulty<1)or(difficulty>3):
            raise NameError("Difficulty must be >=1 and <=3")
        
        self.difficulty = difficulty
        self.random = random
        
        self.xmin = -25.
        self.xmax = 25.
        self.xstep = 0.05
        self.ymin = -25.
        self.ymax = 25.
        self.ystep = 0.05
        
        
        
        if (random):
            xwidth = np.abs(self.xmax - self.xmin)
            ywidth = np.abs(self.ymax - self.ymin)
            self.mu1 = [ 0.6*(xwidth*np.random.rand()-xwidth/2.) , 0.6*(ywidth*np.random.rand()-ywidth/2.) ]
            self.mu2 = [ xwidth*np.random.rand()-xwidth/2. , ywidth*np.random.rand()-ywidth/2. ]
            self.mu3 = [ xwidth*np.random.rand()-xwidth/2. , ywidth*np.random.rand()-ywidth/2. ]
        else:
            self.mu1 = [6, 4]
            self.mu2 = [-2, -2]
            self.mu3 = [-7, 10]
        
        self.gaussian1 = multivariate_normal(self.mu1, [[1.0, 0.], [0., 1.]])
        self.gaussian2 = multivariate_normal(self.mu2, [[0.5, 0.3], [0.3, 0.5]])
        self.gaussian3 = multivariate_normal(self.mu3, [[0.8, 0.], [0., 0.8]])
        
        self.weight1 = 10000
        self.weight2 = 1
        self.weight3 = 1E-8
        
        
        self.mu = [self.mu1]
        if (difficulty>1):
            self.mu.append(self.mu2)
            if (difficulty>2):
                self.mu.append(self.mu3)
        
        self.distribution = [self.gaussian1]
        if (difficulty>1):
            self.distribution.append(self.gaussian2)
            if (difficulty>2):
                self.distribution.append(self.gaussian3)
                
        self.weight = [self.weight1]
        if (difficulty>1):
            self.weight.append(self.weight2)
            if (difficulty>2):
                self.weight.append(self.weight3)


    
    # -------------------------------------------------------------------------
    def value(self,pos):  # (pos = [x,y]) 
    # -------------------------------------------------------------------------
        sumval = 0.
        
        for i in range(self.difficulty):
            sumval += self.weight[i]*self.distribution[i].pdf(pos)
        
        return np.fmax(310.+np.log10(sumval), -10.)


    # -------------------------------------------------------------------------
    def plot(self,noFigure=None,fig=None,ax=None, colorbar=True):
    # -------------------------------------------------------------------------
        x, y = np.mgrid[self.xmin:self.xmax:self.xstep, self.ymin:self.ymax:self.ystep]
        pos = np.dstack((x, y))
        potentialFieldForPlot = self.value(pos)
        
        if (fig==None):
            if (noFigure==None):
                noFigure=1
            fig = plt.figure(noFigure)
        if (ax==None):
            ax = fig.add_subplot(111)
        cs = ax.contourf(x, y, potentialFieldForPlot, 20, cmap='BrBG')
        #cs = ax.contour(x, y, potentialFieldForPlot, 10, cmap='BrBG')
        
        if (colorbar):
            fig.colorbar(cs)
        
        return fig, ax

    # -------------------------------------------------------------------------
    def grad(self, pos1, pos2):
    # -------------------------------------------------------------------------
        g = (self.value(pos2) - self.value(pos1)) / np.linalg.norm(pos2-pos1)
        grad =  np.array([g*(pos1[0] - pos2[0]), g*(pos1[1] - pos2[1])])
        return grad
    
    
    # -------------------------------------------------------------------------    
    def meanGrad(self,point, epsilon=1.0):
    # -------------------------------------------------------------------------
        p1 = point.copy()
        p1n = []
        # neighborhood
        p1n.append(p1 + np.array([-epsilon,epsilon]))
        p1n.append(p1 + np.array([epsilon,epsilon]))
        p1n.append(p1 + np.array([epsilon,-epsilon]))
        p1n.append(p1 + np.array([-epsilon,-epsilon]))
        
        
        p1nGrad = []
        meanGrad = np.array([0,0])
        for pt in p1n:
            ptGrad = self.grad(p1, pt)
            
            meanGrad[0] += ptGrad[0]
            meanGrad[1] += ptGrad[1]
            
            p1nGrad.append(ptGrad)
        

        meanGrad = -meanGrad/4.
        
        return meanGrad


    # -------------------------------------------------------------------------    
    def plotQuiverMeanGrad(self, point, epsilon, ax):
    # -------------------------------------------------------------------------
        p1 = point.copy()
        p1n = []
        # neighborhood
        p1n.append(p1 + np.array([-epsilon,epsilon]))
        p1n.append(p1 + np.array([epsilon,epsilon]))
        p1n.append(p1 + np.array([epsilon,-epsilon]))
        p1n.append(p1 + np.array([-epsilon,-epsilon]))
        
        
        p1nGrad = []
        meanGrad = np.array([0,0])
        for pt in p1n:
            ptGrad = self.grad(p1, pt)
            
            #print(ptGrad)
            
            meanGrad[0] += ptGrad[0]
            meanGrad[1] += ptGrad[1]
            
            p1nGrad.append(ptGrad)
            #plt.quiver(pt[0],pt[1],ptGrad[0],ptGrad[1])
        
        #print(meanGrad)
        meanGrad = -meanGrad/4.
        ax.quiver(pt[0],pt[1],meanGrad[0],meanGrad[1])
        
        ax.text(pt[0]+epsilon,pt[1],  '%.2f' % (180./np.pi*np.arctan2(meanGrad[1],meanGrad[0])) )
    
        return meanGrad


# ======================== END OF CLASS Potential =============================




# ========================== MAIN =============================================
if __name__=='__main__':
# =============================================================================
    
    plt.close()
    
    pot = Potential(difficulty=3, random=True)
    
    
    fig2, ax2 = pot.plot(2)
    

    print(pot.value(pot.mu1))
    print(pot.value(pot.mu2))
    print(pot.value(pot.mu3))
    
    
    
    epsilon = 1.0
    
    
    for xx in np.arange(-20,25,5):
        for yy in np.arange(-20,25,5):
            point = np.array([xx,yy])
            pot.plotQuiverMeanGrad(point, epsilon,ax2)
    
    v=pot.meanGrad(np.array([10,10]), epsilon)
    print(v)
    print(np.arctan2(v[1],v[0])*180./np.pi)    
    
    
    
    pot.plot(1)
    
    
    plt.show()
