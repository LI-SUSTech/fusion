import numpy as np
import matplotlib.pyplot as plt

# Load the trajectory data from the files
def load_trajectory(file_path):
    with open(file_path, 'r') as file:
        data = []
        for line in file:
            line = line.strip().split()
            if line:
                timestamp = float(line[0])
                values = [float(v) for v in line[1:]]
                data.append(values)
        return np.array(data)

# Load the trajectories from both files
trajectory1 = load_trajectory('synchronized_trajectory_openvslam.txt')
trajectory2 = load_trajectory('synchronized_trajectory_orb.txt')

# Calculate the average trajectory
average_trajectory = (trajectory1 + trajectory2) / 2

# Extract the X and Y coordinates for 2D plotting
x1, y1 = trajectory1[:, 0], trajectory1[:, 1]
x2, y2 = trajectory2[:, 0], trajectory2[:, 1]
x_avg, y_avg = average_trajectory[:, 0], average_trajectory[:, 1]
output_file_path = 'average_trajectory.txt'
np.savetxt(output_file_path, average_trajectory, fmt='%.8f')
# Plotting the trajectories
plt.figure(figsize=(10, 8))
plt.plot(x1, y1, label='Trajectory 1 (OpenVSLAM)', color='blue', linestyle='--')
plt.plot(x2, y2, label='Trajectory 2 (ORB)', color='green', linestyle='--')
plt.plot(x_avg, y_avg, label='Average Trajectory', color='red', linewidth=2)
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.title('2D Trajectory Comparison')
plt.legend()
plt.grid(True)

# Save the plot to a file
output_plot_path = 'trajectory_comparison_plot.png'
plt.savefig(output_plot_path)

# Show the plot
plt.show()

output_plot_path
