import numpy as np
import matplotlib.pyplot as plt

def read_trajectory(file_path):
    trajectory = []
    with open(file_path, 'r') as file:
        for line in file:
            data = line.strip().split()
            if len(data) < 8:
                continue  # Skip lines with insufficient data
            timestamp = float(data[0])
            position = np.array([float(data[1]), float(data[2]), float(data[3])])
            orientation = np.array([float(data[4]), float(data[5]), float(data[6]), float(data[7])])
            trajectory.append((timestamp, position, orientation))
    return trajectory

def kalman_filter(trajectory1, trajectory2):
    n = min(len(trajectory1), len(trajectory2))  # Use the minimum length to avoid index errors
    fused_trajectory = []

    # Initialize state and covariance matrices
    x = np.zeros(3)  # State vector: [tx, ty, tz]
    P = np.eye(3)    # Covariance matrix

    # Process and measurement noise (these should be tuned for your specific problem)
    Q = np.eye(3) * 0.1
    R = np.eye(3) * 0.1

    for i in range(n):
        # Time update (prediction)
        x = x  # Assuming constant velocity model, no change in state
        P = P + Q  # Increase uncertainty

        # Measurement update (correction)
        timestamp = trajectory1[i][0]
        z1 = trajectory1[i][1]
        z2 = trajectory2[i][1]
        z = (z1 + z2) / 2  # Combine measurements (you can use a different strategy here)

        y = z - x  # Measurement residual
        S = P + R  # Residual covariance
        K = P @ np.linalg.inv(S)  # Kalman gain

        x = x + K @ y  # Update state estimate
        P = (np.eye(3) - K) @ P  # Update covariance estimate

        fused_trajectory.append((timestamp, x.copy()))
    
    return fused_trajectory

def save_trajectory(fused_trajectory, file_path):
    with open(file_path, 'w') as file:
        for pose in fused_trajectory:
            file.write(' '.join(map(str, pose)) + '\n')

def plot_trajectories(trajectory1, trajectory2, fused_trajectory):
    # Extract positions for plotting
    pos1 = np.array([pose[1] for pose in trajectory1])
    pos2 = np.array([pose[1] for pose in trajectory2])
    fused_pos = np.array([pose[1] for pose in fused_trajectory])

    plt.figure()
    plt.plot(pos1[:, 0], pos1[:, 1], label='trajectory_openvslam')
    plt.plot(pos2[:, 0], pos2[:, 1], label='trajectory_orb')
    plt.plot(fused_pos[:, 0], fused_pos[:, 1], label='Fused Trajectory', linestyle='--')
    plt.legend()
    plt.xlabel('X position')
    plt.ylabel('Y position')
    plt.title('Comparison of Trajectories')
    plt.show()

def main():
    # Read the trajectories
    trajectory1 = read_trajectory('synchronized_trajectory_openvslam.txt')
    trajectory2 = read_trajectory('synchronized_trajectory_orb.txt')

    # Apply Kalman filter to fuse the trajectories
    fused_trajectory = kalman_filter(trajectory1, trajectory2)

    # Save the fused trajectory
    save_trajectory(fused_trajectory, 'fused_trajectory.txt')

    # Plot the trajectories for comparison
    plot_trajectories(trajectory1, trajectory2, fused_trajectory)

if __name__ == '__main__':
    main()
