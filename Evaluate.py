import numpy as np


class evaluate:
    def __init__(self):
        pass

    def path_length(self, path):
        """
        計算給定路徑的總長度。

        參數:
        path -- 由(x, y)坐標對組成的路徑列表。

        返回:
        總長度 -- 路徑的總長度。
        """
        total_length = 0
        for i in range(1, len(path)):
            total_length += ((path[i][0] - path[i - 1][0]) ** 2 + (path[i][1] - path[i - 1][1]) ** 2) ** 0.5
        return total_length

    def path_smoothness(self, path):
        """
        计算路径的平滑度，通过计算路径上连续转向角度变化的标准差来评估。

        参数:
        path -- 由(x, y)坐标对组成的路径列表。

        返回:
        平滑度 -- 路径平滑度的评分，标准差的值（值越小，路径越平滑）。
        """
        angle_changes = []
        for i in range(1, len(path) - 1):
            # 计算每三个连续点形成的两个向量
            vec1 = (path[i][0] - path[i - 1][0], path[i][1] - path[i - 1][1])
            vec2 = (path[i + 1][0] - path[i][0], path[i + 1][1] - path[i][1])

            # 计算两向量的单位向量
            unit_vec1 = vec1 / np.linalg.norm(vec1)
            unit_vec2 = vec2 / np.linalg.norm(vec2)

            # 计算两单位向量的夹角（通过点积）
            dot_product = np.dot(unit_vec1, unit_vec2)
            # 防止计算误差导致的值略微超出[-1, 1]范围
            dot_product = np.clip(dot_product, -1.0, 1.0)
            angle_change = np.arccos(dot_product)

            # 将夹角（弧度）添加到列表中
            angle_changes.append(angle_change)

        # 计算所有角度变化的标准差
        smoothness_score = np.std(angle_changes)
        return smoothness_score

    def exploration_rate_with_obstacles(self, map_size, grid_cell_size, obstacles, path, perception_radius, robot_params):
        def initialize_grid_map(map_size, grid_cell_size):
            """初始化网格地图，所有单元格默认未探索"""
            rows, cols = int(map_size[1] / grid_cell_size), int(map_size[0] / grid_cell_size)
            return np.zeros((rows, cols))

        def mark_obstacles(grid_map, obstacles, grid_cell_size):
            """标记障碍物所在的网格单元"""
            for ox, oy, radius in obstacles:
                row, col = int(oy / grid_cell_size), int(ox / grid_cell_size)
                safety_distance_cells = int(radius / grid_cell_size) + int(robot_params['safety_distance'] / grid_cell_size)
                for r in range(-safety_distance_cells, safety_distance_cells + 1):
                    for c in range(-safety_distance_cells, safety_distance_cells + 1):
                        if 0 <= row + r < grid_map.shape[0] and 0 <= col + c < grid_map.shape[1]:
                            grid_map[row + r, col + c] = -1  # -1 表示障碍物

        def calculate_distance(point1, point2):
            """计算两点之间的欧氏距离"""
            return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

        def mark_explored(grid_map, path, perception_radius, grid_cell_size):
            """根据路径和感知范围标记已探索的网格单元"""
            for px, py in path:
                for row in range(grid_map.shape[0]):
                    for col in range(grid_map.shape[1]):
                        cell_center = (col * grid_cell_size + grid_cell_size / 2,
                                       row * grid_cell_size + grid_cell_size / 2)
                        if calculate_distance((px, py), cell_center) <= perception_radius and grid_map[
                            row, col] != -1:
                            grid_map[row, col] = 1  # 1 表示已探索

        """评估考虑障碍物的探索率"""
        grid_map = initialize_grid_map(map_size, grid_cell_size)
        mark_obstacles(grid_map, obstacles, grid_cell_size)
        mark_explored(grid_map, path, perception_radius, grid_cell_size)

        explored_cells = np.sum(grid_map == 1)
        total_cells = np.sum(grid_map >= 0)  # 总可探索单元格不包括障碍物
        exploration_rate = (explored_cells / total_cells) * 100 if total_cells > 0 else 0
        return exploration_rate
