# ============================================================
# Defining Functions for Magnet and Sensor Positions
# ===========================================================

import numpy as np
import cupy as cp
# GPU

def magnetring(n_magnets, init_angles, radius=40):
    # Upload static data to GPU
    positions_cpu = np.array([
        (radius*np.cos(2*np.pi*i/n_magnets), radius*np.sin(2*np.pi*i/n_magnets), 0.0)
        for i in range(n_magnets)
    ], dtype=np.float64) 

    # Initial orientations of the magnets
    Ry90 = cp.array([[0.0, 0.0, 1.0],
                        [0.0, 1.0, 0.0],
                        [-1.0, 0.0, 0.0]], dtype=cp.float64)

    init_rot_batch = cp.zeros((n_magnets, 3, 3), dtype=cp.float64)
    init_rot_batch[:, 0, 0] = cp.cos(init_angles)
    init_rot_batch[:, 0, 1] = -cp.sin(init_angles)
    init_rot_batch[:, 1, 0] = cp.sin(init_angles)
    init_rot_batch[:, 1, 1] = cp.cos(init_angles)
    init_rot_batch[:, 2, 2] = 1.0

    init_orientations_gpu = cp.matmul(init_rot_batch, Ry90)  
    return positions_cpu, init_orientations_gpu


def sensorring(num_points, radius=20):
    # Create points of interest for the fitness function 
    points_of_interest = np.zeros((num_points, 3), dtype=np.float64)
    for i in range(num_points):
        angle = 2 * np.pi * i/num_points
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        points_of_interest [i] = (x, y, 0.0)
    return points_of_interest