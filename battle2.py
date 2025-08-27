import numpy as np
from math import cos, sin, tan, pi, atan2, acos
from random import random, uniform, randint
from gym import spaces
import pygame
import rendering
import pyglet
from pyglet import image
import rendering2


class Model(object):
    def __init__(self, args):
        super(Model, self).__init__()
        self.id = None
        self.enemy = None
        self.size = 0.2
        self.color = None
        self.pos = np.zeros(2)
        self.speed = 0
        self.yaw = 0
        self.death = False
        self.attack_range = 0
        self.attack_angle = 0
        self.sensor_range_l = 0
        self.sensor_range_w = 0
        self.sensor_angle = 0


class Battle(object):
    """为避免神经网络输入数值过大，采用等比例缩小模型"""

    def __init__(self, args, X_range, Y_range, obsCenter, obsR):
        super(Battle, self).__init__()
        self.args = args
        self.dt = 1
        self.t = 0
        self.render_geoms = []
        self.render_geoms_xform = []
        self.render_geoms_colors = []  # 新增：存储几何对象的颜色
        self.static_geoms = []
        self.static_geoms_xform = []
        self.num_CARs = args.num_RCARs + args.num_BCARs
        self.num_cars = args.num_Bcars
        self.num_RCARs = args.num_RCARs
        self.num_BCARs = args.num_BCARs
        self.num_Bcars = args.num_Bcars
        self.CARs = [Model(args) for _ in range(self.num_CARs)]
        self.cars = [Model(args) for _ in range(self.num_cars)]
        self.RCARs = []
        self.BCARs = []
        self.Bcars = []
        for i, CAR in enumerate(self.CARs):
            CAR.id = i
            if i < args.num_BCARs:
                CAR.enemy = False
                CAR.color = np.array([0, 0, 1])
                CAR.attack_range = args.attack_range_B
                CAR.attack_angle = args.attack_angle_BR
                self.BCARs.append(CAR)
            elif i < args.num_RCARs + args.num_BCARs:
                CAR.enemy = True
                CAR.color = np.array([1, 0, 0])
                CAR.attack_range = args.attack_range_R
                CAR.attack_angle = args.attack_angle_BR
                self.RCARs.append(CAR)
        for i, car in enumerate(self.cars):
            car.id = i
            car.enemy = False
            car.color = np.array([0, 0, 1])
            car.sensor_range_l = args.sensor_range_B_l
            car.sensor_range_w = args.sensor_range_B_w
            car.sensor_angle = args.sensor_angle_B
            self.Bcars.append(car)
        self.sensor_range_l = args.sensor_range_B_l
        self.sensor_range_w = args.sensor_range_B_w
        self.viewer = None
        self.action_space = []
        self.x_max, self.y_max = X_range, Y_range
        self.obstacle = obsCenter
        self.obsR = obsR
        self.max_retained_steps = 1000
        self.step_geoms_indices = []
        self.render_static_obstacles()
        self.reset()

    def reset(self):
        self.t = 0
        random_side = randint(0, 1)
        for i, CAR in enumerate(self.CARs):
            CAR.being_attacked = False
            CAR.death = False
        for i, car in enumerate(self.cars):
            car.being_attacked = False
            car.death = False
        self.render_geoms_colors = []  # 重置颜色列表

    def render(self, pos, vel, fire_car, flag_car, HP_index, HP_num, missle_index, missle_num, task, mode='rgb_array'):
        pos_copy = np.copy(pos)
        vel_copy = np.copy(vel)

        if self.viewer is None:
            self.viewer = rendering.Viewer(900, 480)
            pygame.init()

        step_start_index = len(self.render_geoms)
        new_geoms = []
        new_geoms_xform = []
        new_geoms_colors = []  # 新增：存储新几何对象的颜色
        geoms_per_car = []  # 每个车的几何对象数量

        for i, CAR in enumerate(self.CARs):
            start_idx = len(new_geoms)
            if flag_car[i] == 1:
                CAR.color = np.array([0, 0, 0])
            xform = rendering.Transform()
            car_parts = rendering.make_CAR(CAR.size)
            car_color = np.array(CAR.color) * 0.7
            for x in car_parts:
                x.set_color(*car_color, 0.7)
                x.add_attr(xform)
                new_geoms.append(x)
                new_geoms_xform.append(xform)
                new_geoms_colors.append([*car_color, 0.7])
            cockpit = rendering.make_rectangle(width=CAR.size * 0.8, height=CAR.size * 0.4, filled=True)
            cockpit_color = np.array(CAR.color) * 0.8
            cockpit.set_color(*cockpit_color, 0.9)
            cockpit.add_attr(xform)
            cockpit.add_attr(rendering.Transform(translation=(0, CAR.size * 0.2)))
            new_geoms.append(cockpit)
            new_geoms_xform.append(xform)
            new_geoms_colors.append([*cockpit_color, 0.9])
            sector = rendering.make_circle(radius=0.1)
            sector_color = np.array(CAR.color)
            sector.set_color(*sector_color, 0.8)
            sector.add_attr(xform)
            new_geoms.append(sector)
            new_geoms_xform.append(xform)
            new_geoms_colors.append([*sector_color, 0.8])
            if not CAR.enemy:  # 仅为蓝车（BCARs）绘制任务指示器
                if task[i] == -3:
                    color_temp = np.array([0.6, 0.6, 0.6])  # 柔和灰色
                elif task[i] == -2:
                    color_temp = np.array([1.0, 0.5, 0.0])  # 鲜艳橙色
                elif task[i] == -1:
                    color_temp = np.array([0.7, 0.3, 0.9])  # 深紫色
                elif task[i] == 0:
                    color_temp = np.array([0.0, 0.8, 0.8])  # 青绿色
                sector = rendering.make_circle(radius=0.5)
                sector.set_color(*color_temp, 0.5)
                sector.add_attr(xform)
                new_geoms.append(sector)
                new_geoms_xform.append(xform)
                if flag_car[i] == 0:
                    new_geoms_colors.append([*color_temp, 0.5])
                else:
                    new_geoms_colors.append([*color_temp, 0])
            geoms_per_car.append(len(new_geoms) - start_idx)

        # for i, CAR in enumerate(self.CARs):
        #     xform = rendering.Transform()
        #     if fire_car[i] >= 0:
        #         start = pos_copy[i][0:2]
        #         end = pos_copy[int(fire_car[i])][0:2]
        #         arrow_line = rendering.Line(start, end, dashed=True, linewidth=3)
        #         arrow_color = np.array(CAR.color)
        #         arrow_line.set_color(*arrow_color, 1)
        #         arrow_line.add_attr(xform)
        #         new_geoms.append(arrow_line)
        #         new_geoms_xform.append(xform)
        #         new_geoms_colors.append([*arrow_color, 1])

        self.render_geoms.extend(new_geoms)
        self.render_geoms_xform.extend(new_geoms_xform)
        self.render_geoms_colors.extend(new_geoms_colors)
        self.step_geoms_indices.append((step_start_index, len(self.render_geoms)))

        if len(self.step_geoms_indices) > self.max_retained_steps:
            start_idx, _ = self.step_geoms_indices.pop(0)
            self.render_geoms = self.render_geoms[start_idx:]
            self.render_geoms_xform = self.render_geoms_xform[start_idx:]
            self.render_geoms_colors = self.render_geoms_colors[start_idx:]
            self.step_geoms_indices = [(start - start_idx, end - start_idx) for start, end in self.step_geoms_indices]

        # 调整历史步骤的透明度
        decay_factor = 0.96  # 指数衰减因子
        for step_idx, (start, end) in enumerate(self.step_geoms_indices):
            if step_idx < len(self.step_geoms_indices) - 1:  # 跳过当前步骤
                alpha_scale = decay_factor ** (len(self.step_geoms_indices) - 1 - step_idx)
                for i in range(start, end):
                    geom = self.render_geoms[i]
                    current_color = self.render_geoms_colors[i]
                    new_alpha = current_color[3] * alpha_scale
                    geom.set_color(current_color[0], current_color[1], current_color[2], new_alpha)

        self.viewer.geoms = []
        for geom in self.static_geoms:
            self.viewer.add_geom(geom)
        for geom in self.render_geoms:
            self.viewer.add_geom(geom)

        for i, CAR in enumerate(self.CARs):
            xform = rendering.Transform()
            if flag_car[i] != 1 and HP_index[i] != 0:
                health_bar_width = 0.5
                health_bar_height = 0.1
                max_health = HP_num
                num_cells = HP_num
                health_xform = rendering.Transform()
                for j in range(num_cells):
                    x_offset = j * health_bar_width - 0.75
                    health_bar = rendering.FilledPolygon([
                        (x_offset, 0),
                        (x_offset + health_bar_width, 0),
                        (x_offset + health_bar_width, health_bar_height),
                        (x_offset, health_bar_height)
                    ])
                    if HP_index[i] == 0:
                        health_bar.set_color(1, 1, 1, 0)
                    elif j < HP_index[i] / (max_health / num_cells):
                        health_bar.set_color(1, 0.5, 0, 0.9)
                    else:
                        health_bar.set_color(0.3, 0.3, 0.3, 0.5)
                    health_bar.add_attr(health_xform)
                    health_xform.set_translation(pos_copy[i][0] - 0.5, pos_copy[i][1] + CAR.size + 0.5)
                    self.viewer.add_geom(health_bar)
                health_border = rendering.make_rectangle(width=health_bar_width * num_cells, height=health_bar_height,
                                                         filled=False)
                health_border.set_color(0, 0, 0, 0.8)
                health_border.add_attr(health_xform)
                self.viewer.add_geom(health_border)
                missle_bar_width = 0.5
                missle_bar_height = 0.1
                max_missle = missle_num
                num_cells = int(missle_num)
                missle_xform = rendering.Transform()
                for j in range(num_cells):
                    x_offset = j * missle_bar_width - 0.25
                    missle_bar = rendering.FilledPolygon([
                        (x_offset, 0),
                        (x_offset + missle_bar_width, 0),
                        (x_offset + missle_bar_width, missle_bar_height),
                        (x_offset, missle_bar_height)
                    ])
                    if missle_index[i] == 0:
                        missle_bar.set_color(1, 1, 1, 0)
                    elif j < missle_index[i] / (max_missle / num_cells):
                        missle_bar.set_color(0, 1, 0, 0.9)
                    else:
                        missle_bar.set_color(0.3, 0.3, 0.3, 0.5)
                    missle_bar.add_attr(missle_xform)
                    missle_xform.set_translation(pos_copy[i][0] - 0.5, pos_copy[i][1] + CAR.size + 0.3)
                    self.viewer.add_geom(missle_bar)
                missle_border = rendering.make_rectangle(width=missle_bar_width * num_cells, height=missle_bar_height,
                                                         filled=False)
                missle_border.set_color(0, 0, 0, 0.8)
                missle_border.add_attr(missle_xform)
                self.viewer.add_geom(missle_border)

        self.viewer.set_bounds(-self.x_max, self.x_max, -self.y_max, self.y_max)

        current_idx = step_start_index
        for i, CAR in enumerate(self.CARs):
            for idx in range(geoms_per_car[i]):
                if current_idx < len(self.render_geoms_xform):
                    self.render_geoms_xform[current_idx].set_translation(*pos_copy[i][0:2])
                    if vel_copy[i][1] >= 0 and vel_copy[i][0] >= 0:
                        self.render_geoms_xform[current_idx].set_rotation(
                            np.arctan(vel_copy[i][1] / vel_copy[i][0]))
                    elif vel_copy[i][1] < 0 and vel_copy[i][0] >= 0:
                        self.render_geoms_xform[current_idx].set_rotation(
                            np.arctan(vel_copy[i][1] / vel_copy[i][0]))
                    else:
                        self.render_geoms_xform[current_idx].set_rotation(
                            np.arctan(vel_copy[i][1] / vel_copy[i][0]) + np.pi)
                    current_idx += 1
            if fire_car[i] >= 0:
                current_idx += 1  # 跳过攻击箭头的变换

        self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def render_arrow(self, pos, llm_goal, flag_car, mode='rgb_array'):
        pos_copy = np.copy(pos)
        llm_goal_copy = np.copy(llm_goal)
        new_geoms = []
        new_geoms_xform = []
        new_geoms_colors = []  # 新增：存储新几何对象的颜色
        for i, CAR in enumerate(self.CARs):
            if i < self.num_CARs / 2 and flag_car[i] == 0:
                xform = rendering.Transform()
                start = pos_copy[i][0:2]
                end = llm_goal_copy[i][0:2]
                arrow_line = rendering.Line(start, end, dashed=True)
                arrow_color = [0, 0, 1]
                arrow_line.set_color(*arrow_color, 1)
                arrow_line.add_attr(xform)
                new_geoms.append(arrow_line)
                new_geoms_xform.append(xform)
                new_geoms_colors.append([*arrow_color, 1])

        self.render_geoms.extend(new_geoms)
        self.render_geoms_xform.extend(new_geoms_xform)
        self.render_geoms_colors.extend(new_geoms_colors)
        self.step_geoms_indices.append((len(self.render_geoms) - len(new_geoms), len(self.render_geoms)))

        if len(self.step_geoms_indices) > self.max_retained_steps:
            start_idx, _ = self.step_geoms_indices.pop(0)
            self.render_geoms = self.render_geoms[start_idx:]
            self.render_geoms_xform = self.render_geoms_xform[start_idx:]
            self.render_geoms_colors = self.render_geoms_colors[start_idx:]
            self.step_geoms_indices = [(start - start_idx, end - start_idx) for start, end in self.step_geoms_indices]

        # 调整历史步骤的透明度
        decay_factor = 0.96
        for step_idx, (start, end) in enumerate(self.step_geoms_indices):
            if step_idx < len(self.step_geoms_indices) - 1:  # 跳过当前步骤
                alpha_scale = decay_factor ** (len(self.step_geoms_indices) - 1 - step_idx)
                for i in range(start, end):
                    geom = self.render_geoms[i]
                    current_color = self.render_geoms_colors[i]
                    new_alpha = current_color[3] * alpha_scale
                    geom.set_color(current_color[0], current_color[1], current_color[2], new_alpha)

        self.viewer.geoms = []
        for geom in self.static_geoms:
            self.viewer.add_geom(geom)
        for geom in self.render_geoms:
            self.viewer.add_geom(geom)

        self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def vel2yaw(self, vel):
        if vel[1] >= 0 and vel[0] >= 0:
            return np.arctan(vel[1] / vel[0])
        elif vel[1] < 0 and vel[0] >= 0:
            return np.arctan(vel[1] / vel[0])
        else:
            return np.arctan(vel[1] / vel[0]) + np.pi

    def render_static_obstacles(self, ego_pos=(0, 0), ego_yaw=0, BEV_mode=False):
        import rendering2
        self.static_geoms = []
        self.static_geoms_xform = []
        obstacles = [
            {"pos": (-8.95, -4.55), "width": 0.2, "height": 0.9, "color": (0.4, 0.4, 0.4)},
            {"pos": (-8.95, -2.45), "width": 0.2, "height": 1.0, "color": (0.4, 0.4, 0.4)},
            {"pos": (-8.95, 1.95), "width": 0.2, "height": 0.9, "color": (0.4, 0.4, 0.4)},
            {"pos": (-8.95, 4.15), "width": 0.2, "height": 1.0, "color": (0.4, 0.4, 0.4)},
            {"pos": (-4.95, -4.55), "width": 0.2, "height": 0.9, "color": (0.4, 0.4, 0.4)},
            {"pos": (-4.95, -2.45), "width": 0.2, "height": 1.0, "color": (0.4, 0.4, 0.4)},
            {"pos": (-4.95, 1.95), "width": 0.2, "height": 0.9, "color": (0.4, 0.4, 0.4)},
            {"pos": (-4.95, 4.15), "width": 0.2, "height": 1.0, "color": (0.4, 0.4, 0.4)},
            {"pos": (-8.75, -4.55), "width": 2.0, "height": 0.2, "color": (0.4, 0.4, 0.4)},
            {"pos": (-8.75, -1.65), "width": 2.0, "height": 0.2, "color": (0.4, 0.4, 0.4)},
            {"pos": (-8.75, 1.95), "width": 2.0, "height": 0.2, "color": (0.4, 0.4, 0.4)},
            {"pos": (-8.75, 4.95), "width": 2.0, "height": 0.2, "color": (0.4, 0.4, 0.4)},
            {"pos": (-5.45, -4.55), "width": 0.5, "height": 0.2, "color": (0.4, 0.4, 0.4)},
            {"pos": (-5.45, -1.65), "width": 0.5, "height": 0.2, "color": (0.4, 0.4, 0.4)},
            {"pos": (-5.45, 1.95), "width": 0.5, "height": 0.2, "color": (0.4, 0.4, 0.4)},
            {"pos": (-5.45, 4.95), "width": 0.5, "height": 0.2, "color": (0.4, 0.4, 0.4)},
            {"pos": (4.75, -5.55), "width": 0.2, "height": 1.6, "color": (0.4, 0.4, 0.4)},
            {"pos": (4.75, -2.65), "width": 0.2, "height": 1.7, "color": (0.4, 0.4, 0.4)},
            {"pos": (4.75, 0.85), "width": 0.2, "height": 1.7, "color": (0.4, 0.4, 0.4)},
            {"pos": (4.75, 3.85), "width": 0.2, "height": 1.8, "color": (0.4, 0.4, 0.4)},
            {"pos": (8.75, -5.55), "width": 0.2, "height": 1.6, "color": (0.4, 0.4, 0.4)},
            {"pos": (8.75, -2.65), "width": 0.2, "height": 1.7, "color": (0.4, 0.4, 0.4)},
            {"pos": (8.75, 0.85), "width": 0.2, "height": 1.7, "color": (0.4, 0.4, 0.4)},
            {"pos": (8.75, 3.85), "width": 0.2, "height": 1.8, "color": (0.4, 0.4, 0.4)},
            {"pos": (4.95, -5.55), "width": 1.1, "height": 0.2, "color": (0.4, 0.4, 0.4)},
            {"pos": (4.95, -1.15), "width": 1.1, "height": 0.2, "color": (0.4, 0.4, 0.4)},
            {"pos": (4.95, 0.85), "width": 1.1, "height": 0.2, "color": (0.4, 0.4, 0.4)},
            {"pos": (4.95, 5.45), "width": 1.1, "height": 0.2, "color": (0.4, 0.4, 0.4)},
            {"pos": (7.35, -5.55), "width": 1.4, "height": 0.2, "color": (0.4, 0.4, 0.4)},
            {"pos": (7.35, -1.15), "width": 1.4, "height": 0.2, "color": (0.4, 0.4, 0.4)},
            {"pos": (7.35, 0.85), "width": 1.4, "height": 0.2, "color": (0.4, 0.4, 0.4)},
            {"pos": (7.35, 5.45), "width": 1.4, "height": 0.2, "color": (0.4, 0.4, 0.4)},
            {"pos": (-1.15, -3.55), "width": 0.2, "height": 1.8, "color": (0.4, 0.4, 0.4)},
            {"pos": (-1.15, 1.75), "width": 0.2, "height": 1.9, "color": (0.4, 0.4, 0.4)},
            {"pos": (-0.95, -3.55), "width": 1.0, "height": 0.2, "color": (0.4, 0.4, 0.4)},
            {"pos": (-0.95, 3.45), "width": 1.0, "height": 0.2, "color": (0.4, 0.4, 0.4)},
            {"pos": (1.95, -3.55), "width": 0.7, "height": 0.2, "color": (0.4, 0.4, 0.4)},
            {"pos": (1.95, 3.45), "width": 0.7, "height": 0.2, "color": (0.4, 0.4, 0.4)},
            {"pos": (2.65, -3.55), "width": 0.2, "height": 1.8, "color": (0.4, 0.4, 0.4)},
            {"pos": (2.65, 1.75), "width": 0.2, "height": 1.9, "color": (0.4, 0.4, 0.4)},
        ]

        for obs in obstacles:
            xform = rendering2.Transform()
            if not BEV_mode:
                skew_factor = 0.75
                rect_points = [
                    (0, 0),
                    (obs["width"], 0),
                    (obs["width"] * skew_factor, obs["height"]),
                    (0, obs["height"]),
                ]
            else:
                rect_points = [
                    (0, 0),
                    (obs["width"], 0),
                    (obs["width"], obs["height"]),
                    (0, obs["height"]),
                ]
            shadow_offset = 0.1
            shadow = rendering2.make_polygon([
                (0, 0),
                (obs["width"], 0),
                (obs["width"], obs["height"]),
                (0, obs["height"]),
            ], filled=True)
            shadow.set_color(0.1, 0.1, 0.1, 0.75)
            shadow_xform = rendering2.Transform()
            shadow_xform.set_translation(obs["pos"][0] + shadow_offset, obs["pos"][1] - shadow_offset)
            shadow.add_attr(shadow_xform)
            self.static_geoms.append(shadow)
            self.static_geoms_xform.append(shadow_xform)
            rect = rendering2.make_polygon(rect_points, filled=True)
            rect.set_color(*obs["color"], 0.9)
            rect.add_attr(xform)
            self.static_geoms.append(rect)
            self.static_geoms_xform.append(xform)
            xform.set_translation(*obs["pos"])
            bevel_size = 0.2
            top_bevel = rendering2.make_polygon([
                (0, obs["height"]),
                (obs["width"], obs["height"]),
                (obs["width"] - bevel_size, obs["height"] - bevel_size),
                (bevel_size, obs["height"] - bevel_size),
            ], filled=True)
            top_bevel.set_color(1.0, 1.0, 1.0, 0.9)
            top_bevel.add_attr(xform)
            self.static_geoms.append(top_bevel)
            self.static_geoms_xform.append(xform)
            left_bevel = rendering2.make_polygon([
                (0, 0),
                (bevel_size, bevel_size),
                (bevel_size, obs["height"] - bevel_size),
                (0, obs["height"]),
            ], filled=True)
            left_bevel.set_color(1.0, 1.0, 1.0, 0.9)
            left_bevel.add_attr(xform)
            self.static_geoms.append(left_bevel)
            self.static_geoms_xform.append(xform)
            bottom_bevel = rendering2.make_polygon([
                (0, 0),
                (obs["width"], 0),
                (obs["width"] - bevel_size, bevel_size),
                (bevel_size, bevel_size),
            ], filled=True)
            bottom_bevel.set_color(0.1, 0.1, 0.1, 0.9)
            bottom_bevel.add_attr(xform)
            self.static_geoms.append(bottom_bevel)
            self.static_geoms_xform.append(xform)
            right_bevel = rendering2.make_polygon([
                (obs["width"], 0),
                (obs["width"], obs["height"]),
                (obs["width"] - bevel_size, obs["height"] - bevel_size),
                (obs["width"] - bevel_size, bevel_size),
            ], filled=True)
            right_bevel.set_color(0.1, 0.1, 0.1, 0.9)
            right_bevel.add_attr(xform)
            self.static_geoms.append(right_bevel)
            self.static_geoms_xform.append(xform)
            if BEV_mode:
                cap_offset = 0.4
                cap = rendering2.make_polygon([
                    (0, obs["height"]),
                    (obs["width"], obs["height"]),
                    (obs["width"], obs["height"] + cap_offset),
                    (0, obs["height"] + cap_offset),
                ], filled=True)
                cap.set_color(1.0, 1.0, 1.0, 0.9)
                cap_xform = rendering2.Transform()
                cap_xform.set_translation(obs["pos"][0], obs["pos"][1])
                cap.add_attr(cap_xform)
                self.static_geoms.append(cap)
                self.static_geoms_xform.append(cap_xform)

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None