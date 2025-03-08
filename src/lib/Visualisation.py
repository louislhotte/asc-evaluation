import numpy as np
import matplotlib.pyplot as plt
from simulation import FleetSimulation
from robot import Fleet, si_to_uni
import control_algo_potential

nbOfRobots = 8
fleet = Fleet(nbOfRobots, dynamics='singleIntegrator2D')
for i in range(nbOfRobots):
    fleet.robot[i].state = 20 * np.random.rand(2) - 10

Te = 0.1
simulation = FleetSimulation(fleet, t0=0.0, tf=20.0, dt=Te)

xmin, xmax = -25, 25
ymin, ymax = -25, 25
xgrid = np.linspace(xmin, xmax, 100)
ygrid = np.linspace(ymin, ymax, 100)
X, Y = np.meshgrid(xgrid, ygrid)
dummy_vx, dummy_vy, pot = control_algo_potential.potential_seeking_ctrl(0, 0, fleet.getPosesArray())
Z = np.zeros_like(X)
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        Z[i, j] = pot.value(np.array([[X[i, j], Y[i, j]]]))

fig, ax = plt.subplots(figsize=(6, 5))
cmap = plt.cm.magma
contour = ax.contourf(X, Y, Z, levels=50, cmap=cmap)
cbar = plt.colorbar(contour, ax=ax, label='Potential')
ax.set_xlabel('x', fontsize=12)
ax.set_ylabel('y', fontsize=12)
ax.set_title('Potential Map with Live Robot Movement', fontsize=14, fontweight='bold')
ax.set_xlim(xmin, xmax)
ax.set_ylim(ymin, ymax)
colors = plt.cm.rainbow(np.linspace(0, 1, nbOfRobots))
scatters = []
for i in range(nbOfRobots):
    s = ax.scatter([], [], c=[colors[i]], marker='o', edgecolors='black', s=100, label=f'Robot {i+1}')
    scatters.append(s)
ax.legend(fontsize=10, loc='upper right', frameon=True, shadow=True, fancybox=True)

potential_history = [[] for _ in range(nbOfRobots)]
init_poses = fleet.getPosesArray().copy()
initial_potentials = [pot.value(np.array([init_poses[r, :]])) for r in range(nbOfRobots)]
for r in range(nbOfRobots):
    potential_history[r].append(initial_potentials[r])

plt.ion()
plt.show()

for t in simulation.t:
    robots_poses = fleet.getPosesArray()
    for r in range(nbOfRobots):
        vx, vy, pot = control_algo_potential.potential_seeking_ctrl(t, r, robots_poses)
        fleet.robot[r].ctrl = np.array([vx, vy])
    simulation.addDataFromFleet(fleet)
    fleet.integrateMotion(Te)
    robots_poses = fleet.getPosesArray()
    for r in range(nbOfRobots):
        scatters[r].set_offsets([robots_poses[r, :]])
    if abs(t - round(t)) < Te/2:
        for r in range(nbOfRobots):
            potential_history[r].append(pot.value(np.array([robots_poses[r, :]])))
    plt.pause(0.001)

plt.ioff()
plt.show()

print("Potential history (sampled each second):")
for r in range(nbOfRobots):
    print(f"Robot {r+1}: {potential_history[r]}")