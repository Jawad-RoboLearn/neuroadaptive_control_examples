# A 2-Dof robotic arm where the load at end effector is adaptively control with RBF adaptive control approach.

import numpy as np
import matplotlib.pyplot as plt

# RBF function definition
def radial_basis_fcn(x, centers, beta):
    return np.array([np.exp(-beta * np.linalg.norm(x - c)**2) for c in centers])

# Adaptive control law
def adaptive_control(q, dq, desired_q, desired_dq, what, centers, beta):
    # Nominal control law (e.g., PD control)
    Kp = 10  # Proportional gain
    Kd = 5   # Derivative gain
    un = Kp * (desired_q - q) + Kd * (desired_dq - dq)

    # RBF for adaptation
    theta = rbf(q, centers, beta)
    
    # adaptive control inputs
    ua = np.dot(what, theta)  

    # Total control input
    u = un + ua
    return u

# Weight update rule
def update_weights(what, theta, error, gamma, dt):
    leakage = 0.1
    # Update weights for each basis function
    for j in range(len(what)):
        what[j] += dt * (gamma * (theta[j] * error[0] - leakage * what[j]))
    return what

# main
q = np.array([0.0, 0.0])  # Joint angles
dq = np.array([0.0, 0.0])  # Joint velocities
what = np.zeros(9)  # RBF weights
centers = np.array([[0, 0], [1, 1], [-1, -1], [1, -1], [-1, 1], [0, 1], [0, -1], [1, 0], [-1, 0]])
beta = 0.5
gamma = 50
dt = 0.01
desired_q = np.array([np.pi/4, np.pi/4])  # Desired position
desired_dq = np.array([0.0, 0.0])  # Desired velocity

# Data recording for plotting
time_record = []
q_record = []
dq_record = []
u_record = []

# Simulation loop
for t in np.arange(0, 10, dt):
    time_record.append(t)

    # Calculate the control input
    u = adaptive_control(q, dq, desired_q, desired_dq, what, centers, beta)
    u_record.append(u)

    # Simulate dynamics (simplified)
    q += dq * dt
    dq += u * dt  # Apply torque to update velocity

    # Introduce an external load (change in load)
    F_load = np.random.uniform(-1, 1, size=q.shape)  # Random load
    dq += F_load * dt  # Update velocity with load

    # Calculate error
    error = desired_q - q

    # Update weights
    what = update_weights(what, rbf(q, centers, beta), error, gamma, dt)

    # Record joint angles and velocities
    q_record.append(q.copy())
    dq_record.append(dq.copy())

# Convert recorded data to arrays for plotting
q_record = np.array(q_record)
dq_record = np.array(dq_record)
u_record = np.array(u_record)

# Plotting the results
plt.figure(figsize=(12, 8))

# Plot joint angles
plt.subplot(3, 1, 1)
plt.plot(time_record, q_record[:, 0], label='Joint 1 Angle (rad)', color='blue')
plt.plot(time_record, q_record[:, 1], label='Joint 2 Angle (rad)', color='orange')
plt.axhline(y=np.pi/4, color='green', linestyle='--', label='Desired Angle')
plt.xlabel('Time (s)')
plt.ylabel('Joint Angles (rad)')
plt.title('Joint Angles Over Time')
plt.legend()
plt.grid()

# Plot joint velocities
plt.subplot(3, 1, 2)
plt.plot(time_record, dq_record[:, 0], label='Joint 1 Velocity (rad/s)', color='blue')
plt.plot(time_record, dq_record[:, 1], label='Joint 2 Velocity (rad/s)', color='orange')
plt.xlabel('Time (s)')
plt.ylabel('Joint Velocities (rad/s)')
plt.title('Joint Velocities Over Time')
plt.legend()
plt.grid()

# Plot control inputs
plt.subplot(3, 1, 3)
plt.plot(time_record, u_record[:, 0], label='Control Input Joint 1', color='blue')
plt.plot(time_record, u_record[:, 1], label='Control Input Joint 2', color='orange')
plt.xlabel('Time (s)')
plt.ylabel('Control Input (torque)')
plt.title('Control Inputs Over Time')
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()
