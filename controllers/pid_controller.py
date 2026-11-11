"""
PID controller for the inverted pendulum.

Cascaded structure:
- Inner loop: PD control on pole angle (theta)
- Outer loop: PID control on cart position (x)
"""


class InvertedPendulumPID:
    """
    Cascaded PID controller for the inverted pendulum.

    Inner loop: pole angle (theta) — PD control (faster, more important).
    Outer loop: cart position (x) — PID control (slower, secondary).

    Control law:
        u = kp_theta * theta
          + kd_theta * theta_dot
          + kp_x * x
          + ki_x * integral_x
          + kd_x * x_dot
    """

    def __init__(self, kp_theta, kd_theta, kp_x, ki_x, kd_x, dt):
        # Angle loop gains (inner, primary)
        self.kp_theta = kp_theta
        self.kd_theta = kd_theta

        # Position loop gains (outer, secondary)
        self.kp_x = kp_x
        self.ki_x = ki_x
        self.kd_x = kd_x

        # Simulation timestep (needed for integration)
        self.dt = dt

        # Integral accumulator — persists across calls
        self.integral_x = 0.0

    def update(self, x, x_dot, theta, theta_dot):
        """
        Compute control force given current state.

        Args:
            x: cart position (m)
            x_dot: cart velocity (m/s)
            theta: pole angle (rad)
            theta_dot: pole angular velocity (rad/s)

        Returns:
            Control force (N).
        """
        # Accumulate the integral of cart position error
        self.integral_x += x * self.dt

        # Combine angle loop and position loop
        u = (
            self.kp_theta * theta
            + self.kd_theta * theta_dot
            + self.kp_x * x
            + self.ki_x * self.integral_x
            + self.kd_x * x_dot
        )

        return u
