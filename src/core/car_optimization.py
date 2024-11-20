import casadi as ca
import gc

class Optimization:
    def __init__(self, params, dynamics):
        self.params = params
        self.dynamics = dynamics
        self.define_objective()

    def define_objective(self):
        # Scale the control penalty weights
        acc_penalty_weight = 1e-2
        steer_penalty_weight = 1e-2
        steer_rate_penalty_weight = 1e-1  # New weight for steering rate penalty

        # Calculate the scaled control penalty
        acc_penalty = ca.sumsqr(self.params.U[0, :])
        steer_penalty = ca.sumsqr(self.params.U[1, :])

        # Add a penalty for the rate of change of steering angle
        steer_rate_penalty = ca.sumsqr(self.params.U[1, 1:] - self.params.U[1, :-1])

        # Add a small penalty on velocity and heading angle
        state_penalty_weight = 1e-3
        velocity_penalty = ca.sumsqr(self.params.X[2, :])
        heading_penalty = ca.sumsqr(self.params.X[3, :])

        # Add a penalty for the total distance traveled
        distance_penalty_weight = 1.0  # Adjust this weight to balance path length vs. other objectives
        distance_penalty = 0
        for k in range(self.params.N):
            dx = self.params.X[0, k+1] - self.params.X[0, k]
            dy = self.params.X[1, k+1] - self.params.X[1, k]
            distance_penalty += ca.sqrt(dx**2 + dy**2)

        # Combine all penalty terms
        scaled_control_penalty = (
            acc_penalty_weight * acc_penalty +
            steer_penalty_weight * steer_penalty +
            steer_rate_penalty_weight * steer_rate_penalty +  # Add the steering rate penalty
            state_penalty_weight * (velocity_penalty + heading_penalty)
        )

        # The time T is not scaled, so we can use it directly
        self.params.opti.minimize(
            self.params.T + 
            scaled_control_penalty +
            distance_penalty_weight * distance_penalty
        )

    def cleanup(self):
        del self.params
        del self.dynamics
        gc.collect()