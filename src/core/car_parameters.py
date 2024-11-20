import numpy as np
from casadi import Opti
import gc

class SimulationParameters:
    def __init__(self):
        self.car_length = 4.8  # Car length in meters
        self.car_width = 1.8   # Car width in meters
        self.N = 200
        self.opti = Opti()
        self.X = self.opti.variable(4, self.N + 1)  # 4 state variables: x, y, v, theta
        self.U = self.opti.variable(2, self.N)  # 2 control inputs: a (acceleration), delta (steering angle)
        self.T = self.opti.variable()
        self.use_straight_line_guess = False

        # Define scaling factors
        self.pos_scale = 10
        self.vel_scale = 5.0
        self.angle_scale = 1
        self.acc_scale = 2.0
        self.steer_scale = np.pi/4

        # Define multiple circular boundaries for the car
        self.car_boundaries = [
            #{'center': (1, 0), 'radius': 1.0},  # Front circle
            #{'center': (-1, 0), 'radius': 1.0}, # Rear circle
            {'center': (self.car_length/2,   self.car_width/2), 'radius':  0.10},  # Front left
            {'center': (self.car_length/2, - self.car_width/2-0.25), 'radius':  0.10}, # Front right
            {'center': (-self.car_length/2,  self.car_width/2), 'radius':  0.10},  # Back left
            {'center': (-self.car_length/2, -self.car_width/2-0.25), 'radius':  0.10}, # Back right
            {'center': (0,  self.car_width/2), 'radius':  0.10},  # Middle left
            {'center': (0, -self.car_width/2-0.25), 'radius':  0.10}, #  Middle right
            {'center': ( self.car_length/2, 0), 'radius': 0.10},  # Middle front
            {'center': (-self.car_length/2, 0), 'radius': 0.10}, #  Middle back
        ]

        self._initialize_state_variables()
        self._initialize_control_variables()

    def _initialize_state_variables(self):
        # Scaled state variables
        self.pos_x = self.pos_scale * self.X[0, :]
        self.pos_y = self.pos_scale * self.X[1, :]
        self.v = self.vel_scale * self.X[2, :]
        self.theta = self.angle_scale * self.X[3, :]

    def _initialize_control_variables(self):
        # Scaled control variables
        self.a = self.acc_scale * self.U[0, :]
        self.delta = self.steer_scale * self.U[1, :]

        self.initial_guess_path_x = None
        self.initial_guess_path_y = None

    def unscale_state(self, scaled_state):
        unscaled_state = np.zeros_like(scaled_state)
        unscaled_state[0:2] = self.pos_scale * scaled_state[0:2]
        unscaled_state[2] = self.vel_scale * scaled_state[2]
        unscaled_state[3] = self.angle_scale * scaled_state[3]
        return unscaled_state

    def unscale_control(self, scaled_control):
        unscaled_control = np.zeros_like(scaled_control)
        unscaled_control[0] = self.acc_scale * scaled_control[0]
        unscaled_control[1] = self.steer_scale * scaled_control[1]
        return unscaled_control

    def scale_state(self, unscaled_state):
        scaled_state = np.zeros_like(unscaled_state)
        scaled_state[0:2] = unscaled_state[0:2] / self.pos_scale
        scaled_state[2] = unscaled_state[2] / self.vel_scale
        scaled_state[3] = unscaled_state[3] / self.angle_scale
        return scaled_state

    def scale_control(self, unscaled_control):
        scaled_control = np.zeros_like(unscaled_control)
        scaled_control[0] = unscaled_control[0] / self.acc_scale
        scaled_control[1] = unscaled_control[1] / self.steer_scale
        return scaled_control

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()

    def cleanup(self):
        del self.opti
        del self.X
        del self.U
        del self.T
        del self.pos_x
        del self.pos_y
        del self.v
        del self.theta
        del self.a
        del self.delta
        del self.initial_guess_path_x
        del self.initial_guess_path_y
        gc.collect()