import numpy as np

# Load the trajectory data with flexible parsing
def load_trajectory_flexible(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            # Clean and split the line into parts
            parts = line.strip().split()
            if parts:
                # Convert to floats if possible, ignore otherwise
                try:
                    data.append([float(part.strip('[]')) for part in parts if part.strip('[]')])
                except ValueError:
                    continue
    return np.array(data)

# Load the fused trajectory and ground truth data
fused_trajectory = load_trajectory_flexible('fused_trajectory.txt')
ground_truth = np.loadtxt('gt.csv', delimiter=',')

# Extract the position columns from both trajectories (assuming they are the first three columns after the timestamp)
fused_position = fused_trajectory[:, 1:4]
ground_truth_position = ground_truth[:, 1:4]

# Ensure both trajectories have the same length for comparison
min_length = min(len(fused_position), len(ground_truth_position))
fused_position = fused_position[:min_length]
ground_truth_position = ground_truth_position[:min_length]

# Calculate the error (Euclidean distance) between the fused trajectory and the ground truth
errors = np.linalg.norm(fused_position - ground_truth_position, axis=1)

# Calculate the mean error
mean_error = np.mean(errors)

print(mean_error)
