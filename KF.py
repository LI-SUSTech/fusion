import numpy as np
import matplotlib.pyplot as plt

def read_trajectory(file_path):
    trajectory = []
    with open(file_path, 'r') as file:
        for line in file:
            data = line.strip().split()
            if len(data) < 4:
                continue  # Skip lines with insufficient data
            timestamp = float(data[0])
            position = np.array([float(data[1]), float(data[2]), float(data[3])])
            trajectory.append((timestamp, position))
    return trajectory

def kalman_filter(trajectory1, trajectory2):
    n = min(len(trajectory1), len(trajectory2))  
    fused_trajectory = []

    # Initialize state and covariance matrices
    x = np.zeros(3)  # State vector: [tx, ty, tz]
    P = np.eye(3)    # Covariance matrix

    # Initialize process noise and measurement noise covariances
    Q = np.eye(3) * 0.1
    R1 = np.eye(3) * 0.1  # Initial measurement noise covariance for trajectory 1
    R2 = np.eye(3) * 0.1  # Initial measurement noise covariance for trajectory 2

    for i in range(n):
        # Time update (prediction)
        P = P + Q  # Increase uncertainty due to process noise

        # Measurement update for trajectory 1
        timestamp = trajectory1[i][0]
        z1 = trajectory1[i][1]  # Measurement from trajectory 1
        z2 = trajectory2[i][1]  # Measurement from trajectory 2

        # Compute Kalman gain for trajectory 1
        S1 = P + R1  # Residual covariance
        K1 = P @ np.linalg.inv(S1)  # Kalman gain
        x1 = x + K1 @ (z1 - x)  # Update state estimate using trajectory 1
        P1 = (np.eye(3) - K1) @ P  # Update covariance for trajectory 1

        # Compute Kalman gain for trajectory 2
        S2 = P1 + R2  # Residual covariance
        K2 = P1 @ np.linalg.inv(S2)  # Kalman gain
        x = x1 + K2 @ (z2 - x1)  # Further update state estimate using trajectory 2
        P = (np.eye(3) - K2) @ P1  # Update covariance for trajectory 2

        # Q and R update based on innovation (residual)
        innovation = z2 - x
        Q = Q + 0.01 * (np.outer(innovation, innovation) - Q)
        R2 = R2 + 0.01 * (S2 - R2)

        # Ensure P matrix remains symmetric and non-negative
        P = 0.5 * (P + P.T)
        P = np.clip(P, 1e-12, None)  # Ensure no negative values in P

        # Print P, Q, and R matrices to check if they're being updated
        print(f"Iteration {i + 1}: P matrix\n{P}\n")
        print(f"Iteration {i + 1}: Q matrix\n{Q}\n")
        print(f"Iteration {i + 1}: R2 matrix\n{R2}\n")

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
