import numpy as np
import matplotlib.pyplot as plt
from src.core.car_parameters import SimulationParameters
from src.core.car_dynamics import CarDynamics
from src.core.car_constraints import Constraints
from src.environments.car_obstacles import Obstacles
from src.core.car_initial_guess import InitialGuess
from src.core.car_optimization import Optimization
from src.core.car_solver import Solver
from src.visualization.car_animation import Animation
import logging
import gc

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_simulation(num_runs=1, use_straight_line_guess=False, use_csv_guess=False):
    results = []
    perturbed_initial_guesses = []

    original_waypoints = np.array([
        [4.46, 0.2],
        [3.5, 8.5],
        [4, 14],
        [2, 8],
        [1.72, 8.955]
    ])

    obstacles = None  # Initialize obstacles outside the loop

    for run in range(num_runs):
        logging.info(f"Starting run {run + 1}")

        with SimulationParameters() as params:
            params.use_straight_line_guess = use_straight_line_guess
            dynamics = CarDynamics(params)
            constraints = Constraints(None, params)
            
            # Only create obstacles once
            if obstacles is None:
                obstacles = Obstacles(params, constraints)
            
            # Create Optimization instance first
            opt = Optimization(params, dynamics)
            
            # Use original waypoints for the first run, perturb for subsequent runs
            if run == 0:
                current_waypoints = original_waypoints
            else:
                current_waypoints = original_waypoints + np.random.normal(0, 2, original_waypoints.shape)
            
            perturbed_initial_guesses.append(current_waypoints)
            
            # Store the current waypoints in params for use in animation
            params.initial_guess_waypoints = current_waypoints
            
            # Now pass the opt instance to InitialGuess
            initial_guess = InitialGuess(opt, params, current_waypoints, use_csv=use_csv_guess)
            
            solver = Solver(opt, params)

            result = {
                'pos_x': solver.pos_x_sol,
                'pos_y': solver.pos_y_sol,
                'v': solver.v_sol,
                'theta': solver.theta_sol,
                'a': solver.a_sol,
                'delta': solver.delta_sol,
                'T': solver.T_sol,
                'initial_guess': current_waypoints
            }
            
            results.append(result)

            # Generate animation for the first run
            if run == 0:
                animation = Animation(solver, params, obstacles)
                
                # Add custom cars
                animation.add_custom_car(x=1.72, y=2.985, theta=np.pi/2, length=4.8, width=1.8, car_type='blue')
                animation.add_custom_car(x=1.72, y=14.925, theta=np.pi/2, length=4.8, width=1.8, car_type='green')
                animation.add_custom_car(x=7.21, y=2.985, theta=np.pi/2, length=4.8, width=1.8, car_type='red')
                animation.add_custom_car(x=7.21, y=8.955, theta=np.pi/2, length=4.8, width=1.8, car_type='red')
                animation.add_custom_car(x=7.21, y=14.925, theta=np.pi/2, length=4.8, width=1.8, car_type='red')

                # Add horizontal lines
                animation.add_horizontal_line(y=5.974 , x_start= 0.73, length=1.98, linewidth=1)
                animation.add_horizontal_line(y=11.948, x_start= 0.73, length=1.98, linewidth=1)
                animation.add_horizontal_line(y=5.974 , x_start=6.278, length=1.98, linewidth=1)
                animation.add_horizontal_line(y=11.948, x_start=6.278, length=1.98, linewidth=1)

                # Create the animation
                animation.create_animation()
                animation.cleanup()

            solver.cleanup()
            opt.cleanup()
            initial_guess.cleanup()
            dynamics.cleanup()

        gc.collect()
        logging.info(f"Completed run {run + 1}")

    # Cleanup obstacles after all runs
    if obstacles:
        obstacles.cleanup()

    logging.info("Simulation completed successfully")

if __name__ == "__main__":
    run_simulation(num_runs=1, use_straight_line_guess=False, use_csv_guess=True)