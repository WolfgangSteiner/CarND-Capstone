from pid import PID

GAS_DENSITY = 2.858
ONE_MPH = 0.44704


class Controller(object):
    def __init__(self, *args, **kwargs):
        self.linear_velocity_pid = PID(1.0, 0.1, 0.5, mn=0.0, mx=1.0)
        self.angluar_velocity_pid = PID(10.0, 0.1, 0.5, mn=-0.43, mx=0.43)

    def control(self,
        target_linear_velocity, target_angular_velocity,
        current_linear_velocity, current_angular_velocity,
        dbw_enabled, **kwargs):
        sample_time = 1 / 50.0
        throttle = self.linear_velocity_pid.step(target_linear_velocity - current_linear_velocity, sample_time)
        steer = self.angluar_velocity_pid.step(target_angular_velocity - current_angular_velocity, sample_time)

        return throttle, 0.0, steer
