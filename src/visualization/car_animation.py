import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, Polygon
import matplotlib.image as mpimg
from matplotlib.transforms import Affine2D
import cv2
from tqdm import tqdm
import gc
from scipy.interpolate import CubicSpline
from scipy.signal import butter, filtfilt
from pathlib import Path

class Animation:
    def __init__(self, solver, params, obstacles):
        self.solver = solver
        self.params = params
        self.obstacles = obstacles
        self.custom_cars = []
        self.horizontal_lines = []
        self.background_color = '#252426'  # Background color
        self.text_color = '#E0C0A0'  # Text and axis line color
        self.steering_ratio = 16
        self.max_steering_angle = np.pi/4  # Maximum steering angle in radians

        # Get the project root directory and assets path
        self.project_root = Path(__file__).parent.parent.parent
        self.assets_dir = self.project_root / 'src' / 'visualization' / 'assets'
        
        # Load images using absolute paths
        try:
            self.car_image = mpimg.imread(str(self.assets_dir / 'car_yellow.png'))
            self.car_green = mpimg.imread(str(self.assets_dir / 'car_green.png'))
            self.car_blue = mpimg.imread(str(self.assets_dir / 'car_blue.png'))
            self.car_red = mpimg.imread(str(self.assets_dir / 'car_red.png'))
            self.steering_wheel = mpimg.imread(str(self.assets_dir / 'steering_wheel.png'))
        except FileNotFoundError as e:
            print(f"Error loading images from {self.assets_dir}")
            print(f"Current working directory: {Path.cwd()}")
            print(f"Project root: {self.project_root}")
            print(f"Specific error: {str(e)}")
            raise

    def add_custom_car(self, x, y, theta, length, width, car_type):
        self.custom_cars.append({
            'x': x,
            'y': y,
            'theta': theta,
            'length': length,
            'width': width,
            'type': car_type
        })

    def add_horizontal_line(self, y, x_start, length, linewidth=1):
        self.horizontal_lines.append({
            'y': y,
            'x_start': x_start,
            'length': length,
            'linewidth': linewidth
        })

    def create_car_boundaries(self, x, y, theta):
        boundaries = []
        for boundary in self.params.car_boundaries:
            center_x = x + boundary['center'][0] * np.cos(theta) - boundary['center'][1] * np.sin(theta)
            center_y = y + boundary['center'][0] * np.sin(theta) + boundary['center'][1] * np.cos(theta)
            boundaries.append(plt.Circle((center_x, center_y), boundary['radius'], fill=False, color=self.text_color, alpha=0.2))
        return boundaries

    def create_animation(self, output_filename='output_animation.mp4'):
        # Ensure output directory exists
        output_dir = self.project_root / 'animations'
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / output_filename

        pos_x_sol = self.solver.pos_x_sol
        pos_y_sol = self.solver.pos_y_sol
        theta_sol = self.solver.theta_sol
        delta_sol = self.solver.delta_sol  # Steering angle
        T_sol = self.solver.T_sol

        fps = 60  # Increased fps for smoother animation
        total_frames = int(T_sol * fps *2)
        t_original = np.linspace(0, T_sol, self.params.N + 1)
        t_interpolated = np.linspace(0, T_sol, total_frames)

        # Use cubic spline interpolation for smoother transitions
        cs_x = CubicSpline(t_original, pos_x_sol)
        cs_y = CubicSpline(t_original, pos_y_sol)
        cs_theta = CubicSpline(t_original, theta_sol)
        cs_delta = CubicSpline(t_original[:-1], delta_sol)  # Delta has one less point

        pos_x_interp = cs_x(t_interpolated)
        pos_y_interp = cs_y(t_interpolated)
        theta_interp = cs_theta(t_interpolated)
        delta_interp = cs_delta(t_interpolated)

        # Apply a low-pass filter to remove high-frequency jitter
        def butter_lowpass_filter(data, cutoff, fs, order=5):
            nyq = 0.5 * fs
            normal_cutoff = cutoff / nyq
            b, a = butter(order, normal_cutoff, btype='low', analog=False)
            y = filtfilt(b, a, data)
            return y

        cutoff = 10  # cutoff frequency in Hz
        pos_x_interp = butter_lowpass_filter(pos_x_interp, cutoff, fps)
        pos_y_interp = butter_lowpass_filter(pos_y_interp, cutoff, fps)
        theta_interp = butter_lowpass_filter(theta_interp, cutoff, fps)
        delta_interp = butter_lowpass_filter(delta_interp, cutoff, fps)

        for obs in self.obstacles.obstacles:
            if obs['type'] == 'dynamic':
                obs_pos = [self.obstacles.get_obstacle_position(t, obs['waypoints']) for t in t_interpolated]
                obs['positions'] = [(self.params.pos_scale * float(pos[0]), 
                                     self.params.pos_scale * float(pos[1])) for pos in obs_pos]
            else:
                obs['positions'] = [obs['position']] * total_frames

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (1920, 1080))

        fig = plt.figure(figsize=(19.2, 10.8), facecolor=self.background_color)
        
        # Create main axes for the 2D environment
        ax_main = fig.add_subplot(111)
        ax_main.set_facecolor(self.background_color)
        ax_main.set_xlim(0, 8.99)
        ax_main.set_ylim(0, 17.91)
        ax_main.set_aspect('equal')

        # Create a smaller axes for the steering wheel
        ax_wheel = fig.add_axes([0.75, 0.65, 0.2, 0.2])  # Adjusted size and position
        ax_wheel.set_facecolor(self.background_color)
        ax_wheel.set_aspect('equal')
        ax_wheel.axis('off')

        # Set color for all text elements and axis lines in main axes
        for ax in [ax_main, ax_wheel]:
            ax.xaxis.label.set_color(self.text_color)
            ax.yaxis.label.set_color(self.text_color)
            ax.tick_params(axis='x', colors=self.text_color)
            ax.tick_params(axis='y', colors=self.text_color)
            ax.title.set_color(self.text_color)
            for spine in ax.spines.values():
                spine.set_edgecolor(self.text_color)

        ax_main.set_xlabel('X (m)')
        ax_main.set_ylabel('Y (m)')

        initial_pos = (pos_x_sol[0], pos_y_sol[0])
        end_pos = (pos_x_sol[-1], pos_y_sol[-1])

        if hasattr(self.params, 'initial_guess_path_x') and hasattr(self.params, 'initial_guess_path_y'):
            initial_guess_x = self.params.initial_guess_path_x
            initial_guess_y = self.params.initial_guess_path_y
            ax_main.plot(initial_guess_x, initial_guess_y, 'y--', linewidth=2, alpha=0.4, label='Initial Guess Path')

        boundary_circles = [plt.Circle((0, 0), 0, fill=False, color=self.text_color, alpha=0.2) for _ in self.params.car_boundaries]
        for circle in boundary_circles:
            ax_main.add_artist(circle)

        yellow_car_img = ax_main.imshow(self.car_image, extent=(-self.params.car_length/2, self.params.car_length/2, 
                                                    -self.params.car_length/2, self.params.car_length/2))
        yellow_car_img.set_zorder(10)

        custom_car_imgs = []
        for car in self.custom_cars:
            if car['type'] == 'green':
                img = ax_main.imshow(self.car_green, extent=(-car['length']/2, car['length']/2, 
                                                        -car['length']/2, car['length']/2))
            elif car['type'] == 'red':
                img = ax_main.imshow(self.car_red, extent=(-car['length']/2, car['length']/2, 
                                                        -car['length']/2, car['length']/2))
            elif car['type'] == 'blue':
                img = ax_main.imshow(self.car_blue, extent=(-car['length']/2, car['length']/2, 
                                                       -car['length']/2, car['length']/2))
            img.set_zorder(9)
            custom_car_imgs.append(img)

        path, = ax_main.plot([], [], color='#F0A050', linewidth=2, label='Optimized Path')
        obstacle_plots = []
        obstacle_circles = []

        for obs in self.obstacles.obstacles:
            #color = 'cyan' if obs['type'] == 'dynamic' else 'black'
            obstacle_plot, = ax_main.plot([], [], 'o', markersize=2, color=self.text_color, alpha=0.2)
            obstacle_circle = Circle((0, 0), obs['radius'], fill=True, color=self.text_color, alpha=0.2)
            ax_main.add_patch(obstacle_circle)
            obstacle_plots.append(obstacle_plot)
            obstacle_circles.append(obstacle_circle)

        # Add horizontal lines
        for line in self.horizontal_lines:
            ax_main.plot([line['x_start'], line['x_start'] + line['length']], 
                    [line['y'], line['y']], 
                    color=self.text_color, 
                    linewidth=line['linewidth'])

        # Add steering wheel image to the separate axes
        steering_wheel_img = ax_wheel.imshow(self.steering_wheel, extent=(-1, 1, -1, 1))

        # Add neutral indicator
        neutral_indicator = ax_wheel.add_patch(Rectangle((0, 0.9), 0.02, 0.1, color='white'))
        
        # Adjust the fixed indicator to make it more visible
        fixed_indicator = ax_wheel.add_patch(Rectangle((-0.1, 1.05), 0.2, 0.05, color='yellow'))

        # Add rotation counter and angle indicator text
        rotation_counter = ax_wheel.text(0, -1.2, "", ha='center', va='center', color='white', fontsize=16)
        angle_indicator = ax_wheel.text(0, 1.2, "", ha='center', va='center', color='white', fontsize=16)

        # Add explanatory text
        ax_wheel.text(0, 1.4, "Tire Angle", ha='center', va='center', color='white', fontsize=16)
        ax_wheel.text(0, -1.4, "Steering Wheel Turn", ha='center', va='center', color='white', fontsize=16)
        ax_wheel.text(1.2, 0, "Right", ha='left', va='center', color='white', fontsize=16, rotation=270)
        ax_wheel.text(-1.2, 0, "Left", ha='right', va='center', color='white', fontsize=16, rotation=90)

        def update_frame(frame):
            tr_yellow = Affine2D().rotate(theta_interp[frame]).translate(pos_x_interp[frame], pos_y_interp[frame])
            yellow_car_img.set_transform(tr_yellow + ax_main.transData)

            for i, car in enumerate(self.custom_cars):
                tr_custom = Affine2D().rotate(car['theta']).translate(car['x'], car['y'])
                custom_car_imgs[i].set_transform(tr_custom + ax_main.transData)

            boundary_circles_new = self.create_car_boundaries(pos_x_interp[frame], pos_y_interp[frame], theta_interp[frame])
            for old_circle, new_circle in zip(boundary_circles, boundary_circles_new):
                old_circle.center = new_circle.center
                old_circle.radius = new_circle.radius
                old_circle.set_alpha(0.5)

            path.set_data(pos_x_interp[:frame+1], pos_y_interp[:frame+1])

            for i, obs in enumerate(self.obstacles.obstacles):
                obs_x, obs_y = obs['positions'][frame]
                obstacle_plots[i].set_data([obs_x], [obs_y])
                obstacle_circles[i].center = (obs_x, obs_y)

            # Update steering wheel rotation
            steering_angle = delta_interp[frame]
            steering_wheel_angle = steering_angle * self.steering_ratio
            tr_steering = Affine2D().rotate(steering_wheel_angle)
            steering_wheel_img.set_transform(tr_steering + ax_wheel.transData)
            neutral_indicator.set_transform(tr_steering + ax_wheel.transData)

            # Update rotation counter
            full_rotations = steering_wheel_angle / (2 * np.pi)
            rotation_counter.set_text(f"{full_rotations:.1f}")

            # Update angle indicator
            angle_degrees = np.degrees(steering_angle)
            angle_indicator.set_text(f"{angle_degrees:.1f}°")

            fig.canvas.draw()
            img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))

            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            return img_bgr

        for frame in tqdm(range(total_frames), desc="Rendering frames"):
            img_bgr = update_frame(frame)
            out.write(img_bgr)

        out.release()
        plt.close(fig)

        del pos_x_sol, pos_y_sol, theta_sol, T_sol
        del pos_x_interp, pos_y_interp, theta_interp
        del fig, ax_main, ax_wheel, yellow_car_img, custom_car_imgs, path, obstacle_plots, obstacle_circles
        gc.collect()

    def cleanup(self):
        del self.solver
        del self.params
        del self.obstacles
        gc.collect()