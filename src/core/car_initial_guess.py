import numpy as np
import pandas as pd
from ..utils.car_utils import smooth_path_2d
import os
from pathlib import Path

class InitialGuess:
    def __init__(self, opt, params, waypoints, use_csv=False):
        self.opt = opt
        self.params = params
        self.waypoints = waypoints
        self.use_csv = use_csv

        # Get the project root directory
        self.project_root = Path(__file__).parent.parent.parent
        # Define data directory path
        self.data_dir = self.project_root / 'data'

        self.generate_initial_guess()

    def generate_initial_guess(self):
        opti = self.opt.params.opti
        N = self.params.N

        if self.use_csv:
            # Load the previous solution from CSV
            try:
                states_path = self.data_dir / 'car_states.csv'
                controls_path = self.data_dir / 'car_controls.csv'

                states_df = pd.read_csv(states_path)
                controls_df = pd.read_csv(controls_path)

                predefined_pos_x = states_df['pos_x'].values
                predefined_pos_y = states_df['pos_y'].values
                predefined_v = states_df['v'].values
                predefined_theta = states_df['theta'].values
                predefined_a = controls_df['a'].values
                predefined_delta = controls_df['delta'].values

                # Interpolate if necessary
                if len(predefined_pos_x) != N + 1:
                    t = np.linspace(0, 1, len(predefined_pos_x))
                    t_new = np.linspace(0, 1, N + 1)
                    predefined_pos_x = np.interp(t_new, t, predefined_pos_x)
                    predefined_pos_y = np.interp(t_new, t, predefined_pos_y)
                    predefined_v = np.interp(t_new, t, predefined_v)
                    predefined_theta = np.interp(t_new, t, predefined_theta)
                    
                    # Interpolate control inputs
                    t_control = np.linspace(0, 1, len(predefined_a))
                    t_control_new = np.linspace(0, 1, N)
                    predefined_a = np.interp(t_control_new, t_control, predefined_a)
                    predefined_delta = np.interp(t_control_new, t_control, predefined_delta)

                print("Successfully loaded initial guess from CSV files.")
            except FileNotFoundError:
                print("CSV file(s) not found. Falling back to default initial guess.")
                self.use_csv = False
            except Exception as e:
                print(f"Error loading CSV files: {str(e)}. Falling back to default initial guess.")
                self.use_csv = False

        if not self.use_csv:
            if self.params.use_straight_line_guess:
                # Generate straight line initial guess
                start = self.waypoints[0]
                end = self.waypoints[-1]
                predefined_pos_x = np.linspace(start[0], end[0], N + 1)
                predefined_pos_y = np.linspace(start[1], end[1], N + 1)
            else:
                # Use existing waypoint-based initial guess
                predefined_pos_x, predefined_pos_y = smooth_path_2d(self.waypoints, N + 1)

            # Generate default values for velocity and heading
            predefined_v = np.ones(N + 1) * 0.5 * self.params.vel_scale
            predefined_theta = np.zeros(N + 1)
            
            # Generate default values for control inputs
            predefined_a = np.zeros(N)
            predefined_delta = np.zeros(N)

        self.params.initial_guess_path_x = predefined_pos_x.copy()
        self.params.initial_guess_path_y = predefined_pos_y.copy()

        # Scale the initial guess
        scaled_pos_x = predefined_pos_x / self.params.pos_scale
        scaled_pos_y = predefined_pos_y / self.params.pos_scale
        scaled_v = predefined_v / self.params.vel_scale
        scaled_theta = predefined_theta / self.params.angle_scale
        scaled_a = predefined_a / self.params.acc_scale
        scaled_delta = predefined_delta / self.params.steer_scale

        opti.set_initial(self.params.T, 10)  # Time is not scaled
        opti.set_initial(self.params.X[0, :], scaled_pos_x)
        opti.set_initial(self.params.X[1, :], scaled_pos_y)
        opti.set_initial(self.params.X[2, :], scaled_v)
        opti.set_initial(self.params.X[3, :], scaled_theta)

        # Control inputs initial guess
        opti.set_initial(self.params.U[0, :], scaled_a)
        opti.set_initial(self.params.U[1, :], scaled_delta)

    def cleanup(self):
        del self.opt
        del self.params
        del self.waypoints