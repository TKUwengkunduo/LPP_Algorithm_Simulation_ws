import numpy as np

def evaluate_trajectory(x_traj, y_traj, goal, obstacles):
    # 计算到达目标点的距离
    dist_to_goal = np.sqrt((x_traj[-1] - goal[0])**2 + (y_traj[-1] - goal[1])**2)
    
    # 初始化最小障碍物距离为无穷大
    min_dist_to_obstacle = float('inf')
    
    # 遍历轨迹上的每个点
    for x, y in zip(x_traj, y_traj):
        # 遍历每个障碍物
        for ox, oy, radius in obstacles:
            # 计算当前点到障碍物中心的距离，并考虑障碍物的半径
            dist = np.sqrt((x - ox)**2 + (y - oy)**2) - radius
            
            # 更新最小障碍物距离
            min_dist_to_obstacle = min(min_dist_to_obstacle, dist)
    
    # 如果最小距离小于安全距离，则认为轨迹与障碍物发生碰撞
    if min_dist_to_obstacle < 1.5:  # 假设障碍物安全距离为1.5米
        return float('inf')
    
    # 返回到达目标点的距离
    return dist_to_goal
