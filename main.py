"""=================================================================================================
    'x', 'y'        代表機器人在二維座標的當前位置
    'theta'         這是機器人當前的朝向或者說是角度
    'v'             機器人當前的線速度，表示機器人沿其朝向的移動速度。單位：米每秒(m/s)
    'omega(ω)'      機器人當前的角速度，表示機器人的旋轉速度。定義了機器人轉向的快慢，單位:弧度每秒(rad/s)
    'goal'          機器人的目標位置，由目標的x和y座標組成
    'obstacles'     障礙物列表，每個障礙物由其x和y座標表示
    'robot_params'  機器人參數
    'global_path'   全局路徑規劃的路徑
================================================================================================="""

"""=================================================================================================
    考慮項目
        機器人視野大小, 障礙物大小, 機器人速度與加速度限制
    待考量
        動態障礙物,噪音
================================================================================================="""

import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from Evaluate_Trajectory import evaluate_trajectory
from Simulate_Trajectory import simulate_trajectory
from Dynamic_Window_Approach import dynamic_window_approach
from Artificial_Potential_Field_Approach import artificial_potential_field_approach
from Velocity_Obstacle_Approach import velocity_obstacle_approach
from Evaluate import evaluate

if __name__ == '__main__':
    evaluator = evaluate()

    # 假设机器人的感知范围为 perception_range
    perception_range = 5.0  # 以单位长度为例

    # 地图和全局路径规划
    map_size = (50, 50)
    obstacles = [(2, 0, 0.2), (10, 9, 1.4), (20, 16, 1.8), (25, 27, 1.0), (30, 29, 1.2)]  # 障碍物信息
    goal = (40, 40)  # 目标位置
    global_path = [(i, i) for i in range(41)]  # 简化的全局路径

    # 机器人参数
    robot_params = {
        'min_speed': 0,
        'max_speed': 2,
        'max_rotation_speed': np.pi / 4,
        'wheel_base': 0.5,
        'safety_distance': 1.5
    }

    x, y, theta = 0, 0, 45  # 初始位置和姿态
    v, omega = 0, 0  # 初始速度和转向速度

    # 记录开始时间
    start_time = time.time()

    # 模拟机器人移动
    local_path_x, local_path_y = [x], [y]
    for _ in range(600):  # 模拟600个时间步
        if obstacles is None or len(obstacles) == 0:
            obstacles_to_pass = None
        else:
            # 仅考虑距离机器人一定范围内的障碍物
            obstacles_to_pass = [obs for obs in obstacles if np.sqrt((obs[0] - x) ** 2 + (obs[1] - y) ** 2) <= perception_range]

        # 使用不同的路径规划方法
        v, omega = dynamic_window_approach(x, y, theta, v, omega, goal, obstacles, robot_params)
        # v, omega = artificial_potential_field_approach(x, y, theta, goal, obstacles_to_pass, robot_params)
        # v, omega = velocity_obstacle_approach(x, y, theta, v, omega, goal, obstacles, robot_params)

        x, y, theta = simulate_trajectory(x, y, theta, v, omega, 0.1, 0.1)[0][-1], \
                      simulate_trajectory(x, y, theta, v, omega, 0.1, 0.1)[1][-1], \
                      theta + omega * 0.1
        local_path_x.append(x)
        local_path_y.append(y)
        if np.sqrt((x - goal[0]) ** 2 + (y - goal[1]) ** 2) < 1:  # 到达目标
            break

    # 记录花费时间
    spend_time = time.time() - start_time

    # 绘图
    plt.figure(figsize=(10, 10))
    plt.plot(*zip(*global_path), label="Global Path")
    for ox, oy, radius in obstacles:
        # 绘制障碍物
        obstacle_circle = Circle((ox, oy), radius, color='red', fill=True, label="Obstacles" if ox == obstacles[0][0] else "")
        plt.gca().add_patch(obstacle_circle)
        # 绘制安全范围
        safety_circle = Circle((ox, oy), radius + robot_params['safety_distance'], color='red', fill=False, linestyle='--')
        plt.gca().add_patch(safety_circle)

    plt.plot(local_path_x, local_path_y, label="Local Path")
    plt.scatter(*goal, color='green', label="Goal")
    plt.xlim(0, map_size[0])
    plt.ylim(0, map_size[1])
    plt.legend()
    plt.grid(True)
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.title('VO Local Path Planning')
    plt.show()

    # 评估路径
    path = list(zip(local_path_x, local_path_y))  # 假设 local_path_x 和 local_path_y 分别存储了路径的 x 和 y 坐标
    path_length = evaluator.path_length(path)
    print("路径长度:", path_length)

    # 花费时间
    print("花费时间:", spend_time)

    # 路径平滑度
    smoothness_score = evaluator.path_smoothness(path)
    print("路径平滑度:", smoothness_score)

    # 探索率
    grid_cell_size = 0.1  # 网格单元的尺寸
    # obstacles = [(10, 10), (20, 20), (30, 30)]  # 障碍物位置
    perception_radius = 5.0  # 机器人感知范围的半径
    exploration_rate = evaluator.exploration_rate_with_obstacles(map_size, grid_cell_size, obstacles, path,
                                                                 perception_radius, robot_params)

    print(f"探索率: {exploration_rate}%")
