import numpy as np
import gc
import casadi as ca

class Constraints:
    def __init__(self, opt, params):
        self.opt = opt
        self.params = params
        self.apply_constraints()

    def apply_constraints(self):
        self._apply_path_constraints()
        self._apply_boundary_conditions()
        self._apply_time_constraints()

    def _apply_path_constraints(self):
        opti = self.params.opti
        X = self.params.X
        U = self.params.U

        # Position constraints
        opti.subject_to(opti.bounded(0, X[0, :], 8.99 / self.params.pos_scale))
        opti.subject_to(opti.bounded(0, X[1, :], 17.91 / self.params.pos_scale))

        # Velocity constraints
        max_velocity = 20 / self.params.vel_scale
        opti.subject_to(opti.bounded(-max_velocity, X[2, :], max_velocity))

        # Heading angle constraints
        #opti.subject_to(opti.bounded(-np.pi / self.params.angle_scale, X[3, :], np.pi / self.params.angle_scale))

        # Acceleration constraints
        max_acc = 2 / self.params.acc_scale
        opti.subject_to(opti.bounded(-max_acc, U[0, :], max_acc))

        # Steering angle constraints
        max_steer = np.pi/4 / self.params.steer_scale
        opti.subject_to(opti.bounded(-max_steer, U[1, :], max_steer))

    def _apply_boundary_conditions(self):
        opti = self.params.opti
        X = self.params.X

        # Initial conditions
        opti.subject_to(X[0:2, 0] == ca.vertcat(4.46, 0.2) / self.params.pos_scale)
        opti.subject_to(X[2, 0] == 0)
        opti.subject_to(X[3, 0] == (np.pi/2) /self.params.angle_scale )

        # Final conditions
        opti.subject_to(X[0:2, -1] == ca.vertcat(1.72,8.955) / self.params.pos_scale) #8.955
        opti.subject_to(X[2, -1] == 0)
        opti.subject_to(X[3, -1] == (np.pi/2) /self.params.angle_scale)

    def _apply_time_constraints(self):
        self.params.opti.subject_to(self.params.opti.bounded(0, self.params.T, 40))

    def cleanup(self):
        del self.opt
        del self.params
        gc.collect()