from __future__ import annotations
import numpy as np
from potential import Potential

firstCall: bool = True
pot: Potential | None = None

def potential_seeking_ctrl(t: float, robotNo: int, robots_poses: np.ndarray) -> tuple[float, float, Potential]:
    global firstCall, pot
    
    if firstCall:
        pot = Potential(difficulty=3, random=True)
        firstCall = False
    
    x: np.ndarray = robots_poses[:, 0:2]
    grad_x, grad_y = pot.grad(x[robotNo, :])

    K = 1.0 # Gain to make the robot move
    vx: float = K * grad_x
    vy: float = K* grad_y
    
    return vx, vy, pot

def my_control_law(t: float, robotNo: int, robots_poses: np.ndarray) -> tuple[float, float]:
    vx: float = 0.0
    vy: float = 0.0
    return vx, vy
