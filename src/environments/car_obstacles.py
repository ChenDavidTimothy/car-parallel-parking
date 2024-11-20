import numpy as np
from casadi import if_else, logic_and, sin, cos, pi

class Obstacles:
    def __init__(self, params, constraints):
        self.params = params
        self.constraints = constraints
        self.obstacles = [
            #{'type': 'dynamic', 'radius': 2, 'waypoints': np.array([[45, 45, 0], [25, 25, 5], [25, 25, 10], [10, 10, 15]])},
            {'type': 'static', 'radius': 0.9, 'position': (1.72, 2.985 + 1.6)},#2.985 + 1.75
            #{'type': 'static', 'radius': 0.2, 'position': (0.82, 5.385)},
            {'type': 'static', 'radius': 0.2, 'position': (2.62, 5.385)},
            {'type': 'static', 'radius': 0.9, 'position': (1.72, 14.925 - 1.6)},
            {'type': 'static', 'radius': 0.2, 'position': (2.62, 12.525)}
        ]
        self.apply_collision_avoidance()

    def get_obstacle_position(self, t, waypoints):
        t_clamped = if_else(t > waypoints[-1][2], waypoints[-1][2], if_else(t < waypoints[0][2], waypoints[0][2], t))
        
        x, y = waypoints[-1][0], waypoints[-1][1]
        
        for i in range(len(waypoints) - 1):
            t1, t2 = waypoints[i][2], waypoints[i+1][2]
            wp1, wp2 = waypoints[i], waypoints[i+1]
            
            alpha = (t_clamped - t1) / (t2 - t1)
            x = if_else(logic_and(t_clamped >= t1, t_clamped <= t2), (1 - alpha) * wp1[0] + alpha * wp2[0], x)
            y = if_else(logic_and(t_clamped >= t1, t_clamped <= t2), (1 - alpha) * wp1[1] + alpha * wp2[1], y)
        
        # Scale the obstacle position
        x_scaled = x / self.params.pos_scale
        y_scaled = y / self.params.pos_scale
        
        return x_scaled, y_scaled

    def apply_collision_avoidance(self):
        for k in range(self.params.N + 1):
            t_k = k * self.params.T / self.params.N
            car_x = self.params.X[0, k]
            car_y = self.params.X[1, k]
            car_theta = self.params.X[3, k]

            for obs in self.obstacles:
                if obs['type'] == 'static':
                    obs_x = obs['position'][0] / self.params.pos_scale
                    obs_y = obs['position'][1] / self.params.pos_scale
                elif obs['type'] == 'dynamic':
                    obs_x, obs_y = self.get_obstacle_position(t_k, obs['waypoints'])

                # Scale the obstacle radius
                scaled_obs_radius = obs['radius'] / self.params.pos_scale

                # Check collision for each car boundary
                for boundary in self.params.car_boundaries:
                    scaled_boundary_radius = boundary['radius'] / self.params.pos_scale

                    # Calculate boundary center position using CasADi operations
                    cos_theta = cos(car_theta * self.params.angle_scale)
                    sin_theta = sin(car_theta * self.params.angle_scale)

                    boundary_center_x = car_x + (boundary['center'][0] * cos_theta - boundary['center'][1] * sin_theta) / self.params.pos_scale
                    boundary_center_y = car_y + (boundary['center'][0] * sin_theta + boundary['center'][1] * cos_theta) / self.params.pos_scale

                    # Calculate squared distance
                    dx = boundary_center_x - obs_x
                    dy = boundary_center_y - obs_y
                    dist_squared = dx**2 + dy**2

                    # Use squared distance for constraint
                    min_dist_squared = (scaled_boundary_radius + scaled_obs_radius)**2

                    # Add constraint
                    self.params.opti.subject_to(dist_squared >= min_dist_squared)

    def cleanup(self):
        del self.params
        del self.constraints
        self.obstacles.clear()