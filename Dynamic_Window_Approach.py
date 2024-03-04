import numpy as np
from Evaluate_Trajectory import evaluate_trajectory
from Simulate_Trajectory import simulate_trajectory

def dynamic_window_approach(x, y, theta, v, omega, goal, obstacles, robot_params):
    # 設定模擬的時間間隔，這決定了每一步模擬的時間長度
    dt = 0.1
    # 設定預測時間，這是對未來軌跡進行模擬的時間範圍
    predict_time = 0.8
    # 從機器人的速度範圍內產生一系列可能的速度值供後續計算使用
    v_samples = np.linspace(robot_params['min_speed'], robot_params['max_speed'], 20)
    # 從機器人的旋轉速度範圍內產生一系列可能的旋轉速度值供後續計算使用
    omega_samples = np.linspace(-robot_params['max_rotation_speed'], robot_params['max_rotation_speed'], 40)

    # 初始化最佳得分為無窮大，用於後續尋找最小得分（最佳軌跡）
    best_score = float('inf')
    # 初始化最佳速度和旋轉速度為當前速度和旋轉速度
    best_v, best_omega = v, omega
    # 遍歷所有可能的速度和旋轉速度組合
    for v_sample in v_samples:
        for omega_sample in omega_samples:
            # 使用模擬函數計算給定速度和旋轉速度下的預測軌跡
            x_traj, y_traj, _ = simulate_trajectory(x, y, theta, v_sample, omega_sample, dt, predict_time)
            # 評估該軌跡，得到一個得分，評估標準包括到達目標的距離和避開障礙物
            score = evaluate_trajectory(x_traj, y_traj, goal, obstacles)
            # 如果該軌跡的得分更低（更優），則更新最佳速度和旋轉速度
            if score < best_score:
                best_score = score
                best_v, best_omega = v_sample, omega_sample

    # 返回計算出的最佳速度和旋轉速度
    return best_v, best_omega