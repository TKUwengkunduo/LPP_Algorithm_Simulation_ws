"""
    需要花較大心力在調整參數
"""


import numpy as np

def artificial_potential_field_approach(x, y, theta, goal, obstacles, robot_params):
    attr_coeff = 150  # 吸引力系数
    rep_coeff = 150  # 排斥力系数
    rep_range = 5   # 排斥力作用范围

    attr_x = attr_coeff * (goal[0] - x)
    attr_y = attr_coeff * (goal[1] - y)

    rep_x, rep_y = 0, 0

    if obstacles:
        for ox, oy, radius in obstacles:
            d = np.sqrt((x - ox) ** 2 + (y - oy) ** 2) - radius  # 考虑障碍物大小
            if d < rep_range:
                rep_x += rep_coeff * (1/rep_range - 1/max(d, 0.1)) * (x - ox) / d  # 避免除以0
                rep_y += rep_coeff * (1/rep_range - 1/max(d, 0.1)) * (y - oy) / d

    force_x = attr_x + rep_x
    force_y = attr_y + rep_y

    target_angle = np.arctan2(force_y, force_x)
    target_speed = min(robot_params['max_speed'], np.sqrt(force_x ** 2 + force_y ** 2))

    omega = (target_angle - np.deg2rad(theta)) / 0.1
    omega = max(min(omega, robot_params['max_rotation_speed']), -robot_params['max_rotation_speed'])

    return target_speed, omega
