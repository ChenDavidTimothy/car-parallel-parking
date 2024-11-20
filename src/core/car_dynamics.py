import numpy as np
from casadi import vertcat, sin, cos, tan
import gc

class CarDynamics:
    def __init__(self, params):
        self.params = params
        self.f = self.define_dynamics()
        self.integrate_dynamics()

    def define_dynamics(self):
        return lambda x, u: vertcat(
            # Position x
            (x[2] * self.params.vel_scale * cos(x[3] * self.params.angle_scale)) / self.params.pos_scale,
            
            # Position y
            (x[2] * self.params.vel_scale * sin(x[3] * self.params.angle_scale)) / self.params.pos_scale,
            
            # Velocity
            (u[0] * self.params.acc_scale) / self.params.vel_scale,
            
            # Heading angle
            (x[2] * self.params.vel_scale * tan(u[1] * self.params.steer_scale) / self.params.car_length) / self.params.angle_scale
        )

    def integrate_dynamics(self):
        dt = self.params.T / self.params.N
        for k in range(self.params.N):
            k1 = self.f(self.params.X[:, k], self.params.U[:, k])
            k2 = self.f(self.params.X[:, k] + dt/2 * k1, self.params.U[:, k])
            k3 = self.f(self.params.X[:, k] + dt/2 * k2, self.params.U[:, k])
            k4 = self.f(self.params.X[:, k] + dt * k3, self.params.U[:, k])
            x_next = self.params.X[:, k] + dt/6 * (k1 + 2 * k2 + 2 * k3 + k4)
            self.params.opti.subject_to(self.params.X[:, k + 1] == x_next)

    def cleanup(self):
        del self.params
        del self.f
        gc.collect()