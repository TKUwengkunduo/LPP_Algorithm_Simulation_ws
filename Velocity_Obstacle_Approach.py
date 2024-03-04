import numpy as np
from Simulate_Trajectory import simulate_trajectory

def velocity_obstacle_approach(x, y, theta, v, omega, goal, obstacles, robot_params):
    dt = 0.1  # 模擬的時間間隔
    predict_time = 2.0  # 預測時間範圍，用於考慮未來的位置
    safety_distance = robot_params['safety_distance']
    
    # 初始化最佳速度和旋轉速度為當前速度和旋轉速度
    best_v, best_omega = v, omega
    min_distance = float('inf')  # 初始化最小距離為無窮大
    
    # 從機器人的速度範圍內產生一系列可能的速度值供後續計算使用
    v_samples = np.linspace(robot_params['min_speed'], robot_params['max_speed'], 20)
    # 從機器人的旋轉速度範圍內產生一系列可能的旋轉速度值供後續計算使用
    omega_samples = np.linspace(-robot_params['max_rotation_speed'], robot_params['max_rotation_speed'], 40)
    
    for v_sample in v_samples:
        for omega_sample in omega_samples:
            collision = False
            x_pred, y_pred, _ = simulate_trajectory(x, y, theta, v_sample, omega_sample, dt, predict_time)
            
            for ox, oy, radius in obstacles:  # Assuming obstacles include size
                for xp, yp in zip(x_pred, y_pred):
                    # Adjusting for obstacle size in collision detection
                    if np.sqrt((xp - ox) ** 2 + (yp - oy) ** 2) < radius + safety_distance:
                        collision = True
                        break
                if collision:
                    break

            
            if not collision:
                distance_to_goal = np.sqrt((x_pred[-1] - goal[0])**2 + (y_pred[-1] - goal[1])**2)
                if distance_to_goal < min_distance:
                    min_distance = distance_to_goal
                    best_v, best_omega = v_sample, omega_sample


    return best_v, best_omega