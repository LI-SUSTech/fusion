import numpy as np

# Function to manually parse the file and clean the data
def load_trajectory_flexible(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            # Clean and split the line into parts
            parts = line.strip().split()
            # Convert to floats if possible, ignore otherwise
            try:
                data.append([float(part) for part in parts])
            except ValueError:
                continue
    return np.array(data)

# Load the average trajectory data
average_trajectory = load_trajectory_flexible('average_trajectory.txt')
ground_truth = np.loadtxt('gt.csv', delimiter=',')

# Extract relevant columns (assuming first 3 columns are position data)
average_trajectory_position = average_trajectory[:, :3]
ground_truth_position = ground_truth[:, 1:4]  # Adjust as necessary

# Calculate the mean error
min_length = min(len(average_trajectory_position), len(ground_truth_position))
average_trajectory_position = average_trajectory_position[:min_length]
ground_truth_position = ground_truth_position[:min_length]

# Calculate the Euclidean distance between points
errors = np.linalg.norm(average_trajectory_position - ground_truth_position, axis=1)
mean_error = np.mean(errors)

print(f"Mean Error: {mean_error}")
