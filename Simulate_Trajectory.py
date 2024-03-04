import numpy as np

def simulate_trajectory(x, y, theta, v, omega, dt, predict_time):
    x_traj, y_traj = [x], [y]
    for _ in np.arange(0, predict_time, dt):
        x += v * np.cos(theta) * dt
        y += v * np.sin(theta) * dt
        theta += omega * dt
        x_traj.append(x)
        y_traj.append(y)
    return x_traj, y_traj, theta