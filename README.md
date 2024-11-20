# Autonomous Car Parallel Parking Simulation

A Python implementation of optimal trajectory planning for autonomous vehicle parallel parking. This project provides robust path optimization with collision avoidance capabilities and stunning 2D visualizations of the parking maneuver with steering wheel feedback.

![Parking Animation](data/animations/car_animation.gif)

## Features

- **Optimal Trajectory Planning**: Implements Direct Multiple Shooting method for finding optimal parking paths
- **Realistic Vehicle Dynamics**: Uses bicycle model with kinematic constraints
- **Collision Avoidance**: 
  - Handles multiple static obstacles
  - Multi-point vehicle boundary representation
  - Parking space constraints
- **Visualization**: 
  - Dynamic 2D visualization of the parking maneuver
  - Real-time steering wheel animation
  - Path trajectory display
  - Multiple vehicle visualization
- **Data Export/Import**: Save and load optimization results for reproducibility
- **Initial Guess Management**: Support for both straight-line and custom initial path guesses

## Requirements

- Python 3.8+
- CasADi (for optimization)
- NumPy
- Matplotlib
- OpenCV (for animation)
- pandas
- scipy
- tqdm
- FFmpeg (for animation export)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/ChenDavidTimothy/car-parallel-parking.git
cd car-parallel-parking
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
```

3. Install the required Python packages:
```bash
pip install -r requirements.txt
```

4. IPOPT Solver installation:
```bash
https://coin-or.github.io/Ipopt/INSTALL.html
```

5. Install FFmpeg (required for animation export):
```bash
# Windows (using chocolatey)
choco install ffmpeg
```

## Project Structure

```
├── src/
│   ├── core/                    # Core algorithm implementations
│   │   ├── car_parameters.py        # Vehicle and simulation parameters
│   │   ├── car_dynamics.py          # Vehicle dynamics model
│   │   ├── car_optimization.py      # Trajectory optimization
│   │   ├── car_solver.py           # Optimization solver
│   │   ├── car_constraints.py      # Physical and problem constraints
│   │   └── car_initial_guess.py    # Initial trajectory generation
│   │
│   ├── environment/            # Environment-related components
│   │   └── car_obstacles.py        # Obstacle definitions and handling
│   │
│   ├── utils/                  # Utility functions
│   │   └── car__utils.py       # Path planning utilities
│   │
│   └── visualization/          # Visualization tools
│       ├── car_animation.py        # Animation generation
│       └── assets/             # Image assets
│
├── data/                      # Data directory
│   ├── car_states.csv        # Saved vehicle states
│   └── car_controls.csv      # Saved control inputs
│
├── precomputed_initial_guess/        # Data directory
│   ├── car_states.csv        # Precomputed vehicle states
│   └── car_controls.csv      # Precomputed control inputs
└── README.md                 # Project documentation
```

## Usage

1. Basic execution:
```python
python car_main.py
```

⚠️ **IMPORTANT**: Initial Guess Setup
- For using precomputed initial guesses for the specific scenario in this repository:
  1. Copy `car_states.csv` and `car_controls.csv` into the `data/` folder
  2. These files contain optimized trajectories that significantly improve convergence
  3. Without these files, the optimizer will fall back to default initialization

2. Configure vehicle parameters in `src/core/parameters.py`:
```python
class SimulationParameters:
    def __init__(self):
        self.car_length = 4.8  # Car length in meters
        self.car_width = 1.8   # Car width in meters
        self.N = 200           # Number of discretization points
```

3. Adjust optimization parameters in `src/core/optimization.py`:
```python
class Optimization:
    def define_objective(self):
        acc_penalty_weight = 1e-2
        steer_penalty_weight = 1e-2
        distance_penalty_weight = 1.0
```

4. Customize obstacle setup in `src/environment/obstacles.py`:
```python
class Obstacles:
    def __init__(self):
        self.obstacles = [
            {'type': 'static', 'radius': 0.9, 'position': (1.72, 4.585)},
            # Add more obstacles as needed
        ]
```

## Algorithm Details

The parking trajectory optimization:

1. **Initialization**: Generate initial trajectory guess
2. **Optimization**: Minimize objective function considering:
   - Path length
   - Control effort (acceleration and steering)
   - Collision avoidance
3. **Constraints**:
   - Vehicle dynamics (bicycle model)
   - Physical limits (velocity, acceleration, steering angle)
   - Obstacle avoidance
   - Initial and final conditions

## Visualization Features

The 2D visualization includes:
- Dynamic vehicle movement
- Real-time steering wheel animation
- Multiple parked vehicles
- Obstacle boundaries
- Optimized path trajectory
- Initial guess path display
- Parking space layout

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Author

David Timothy - chendavidtimothy@gmail.com
GitHub: [@ChenDavidTimothy](https://github.com/ChenDavidTimothy)