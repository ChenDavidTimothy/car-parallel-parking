import pandas as pd
import casadi as ca
import numpy as np
import logging
import gc
import os
from pathlib import Path

class Solver:
    def __init__(self, opt, params):
        self.opt = opt
        self.params = params
        self.solve()

    def solve(self):
        p_opts = {"expand": True}
        s_opts = {
            "max_iter": 20000,
            "mumps_pivtol": 5e-7,
            "mumps_mem_percent": 50000,
            "linear_solver": "mumps",
            "constr_viol_tol": 1e-7,
            "print_level": 5,
            "nlp_scaling_method": "gradient-based",
            "mu_strategy": "adaptive",
            "check_derivatives_for_naninf": "yes",
            "hessian_approximation": "exact",
            "tol": 1e-8,
        }


        self.opt.params.opti.solver("ipopt", p_opts, s_opts)

        try:
            self.sol = self.opt.params.opti.solve()
        except RuntimeError as e:
            print("Optimization failed. Debugging information:")
            print(f"Error message: {str(e)}")
            
            # Print bounds for all variables
            for i, var in enumerate(self.opt.params.opti.x):
                lb = self.opt.params.opti.debug.value(self.opt.params.opti.lbx[i])
                ub = self.opt.params.opti.debug.value(self.opt.params.opti.ubx[i])
                print(f"Variable {i}: Lower bound = {lb}, Upper bound = {ub}")
            
            # Print all constraint violations
            for i, g in enumerate(self.opt.params.opti.g):
                violation = self.opt.params.opti.debug.value(g)
                print(f"Constraint {i}: Violation = {violation}")
            
            # Print the objective function value
            obj_value = self.opt.params.opti.debug.value(self.opt.params.opti.f)
            print(f"Objective function value: {obj_value}")
            
            raise

        self.extract_solution()
        self.save_solution_to_csv()
        del self.sol
        gc.collect()

    def extract_solution(self):
        # Unscale the solution
        self.pos_x_sol = self.params.pos_scale * self.sol.value(self.opt.params.X[0, :])
        self.pos_y_sol = self.params.pos_scale * self.sol.value(self.opt.params.X[1, :])
        self.v_sol = self.params.vel_scale * self.sol.value(self.opt.params.X[2, :])
        self.theta_sol = self.params.angle_scale * self.sol.value(self.opt.params.X[3, :])
        self.a_sol = self.params.acc_scale * self.sol.value(self.opt.params.U[0, :])
        self.delta_sol = self.params.steer_scale * self.sol.value(self.opt.params.U[1, :])
        self.T_sol = self.sol.value(self.opt.params.T)

    def save_solution_to_csv(self):
        # Get the project root directory
        project_root = Path(__file__).parent.parent.parent
        data_dir = project_root / 'data'

        data_dir.mkdir(parents=True, exist_ok=True)
        states_df = pd.DataFrame({
            'pos_x': self.pos_x_sol,
            'pos_y': self.pos_y_sol,
            'v': self.v_sol,
            'theta': self.theta_sol
        })
        controls_df = pd.DataFrame({
            'a': self.a_sol,
            'delta': self.delta_sol
        })

        # Save to CSV files in the data directory
        states_df.to_csv(data_dir / 'car_states.csv', index=False)
        controls_df.to_csv(data_dir / 'car_controls.csv', index=False)

    def cleanup(self):
        del self.opt
        del self.params
        gc.collect()