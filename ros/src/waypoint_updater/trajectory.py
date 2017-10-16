import numpy as np
from numpy.polynomial import Polynomial

class Trajectory:
    def __init__(self, start_state, end_state, duration, delay=0.0, total_duration=10.0, sample_rate=100.0):
        self.start_state = np.array(start_state)
        self.end_state = np.array(end_state)
        self.duration = duration
        self.delay = delay
        self.total_duration = total_duration
        self.sample_rate = sample_rate
        self.max_jerk = 10.0
        self.max_acceleration = 1.0
        self.max_deceleraion = 5.0


    @staticmethod
    def StoppingTrajectory(start_state, end_state, duration, delay, **kwargs):
        delay = kwargs.get('delay', 0.0)
        total_duration = kwargs.get('total_duration', 10.0)
        sample_rate = kwargs.get('sample_rate', 100.0)

        trajectory = Trajectory(
            start_state, end_state, duration,
            delay=delay, total_duration=total_duration, sample_rate=sample_rate)

        trajectory.max_acceleration = kwargs.get('accel_limit', 1.0)
        trajectory.max_deceleration = kwargs.get('decel_limit', 5.0)
        
        trajectory.polynomial = Trajectory.calc_quintic_polynomial(
            trajectory.state_at_time(delay),
            end_state,
            duration)

        return trajectory


    @staticmethod
    def VelocityKeepingTrajectory(start_state, end_state, duration, delay, **kwargs):
        delay = kwargs.get('delay', 0.0)
        total_duration = kwargs.get('total_duration', 10.0)
        sample_rate = kwargs.get('sample_rate', 100.0)

        trajectory = Trajectory(
            start_state, end_state, duration,
            delay, total_duration=total_duration, sample_rate=sample_rate)

        trajectory.polynomial = Trajectory.calc_quartic_polynomial(
            trajectory.state_at_time(delay),
            end_state,
            duration)

        trajectory.max_acceleration = kwargs.get('accel_limit', 1.0)
        trajectory.max_deceleration = kwargs.get('decel_limit', 5.0)

        # Update the end position based on the polynomial:
        trajectory.end_state[0] = trajectory.polynomial(duration - delay)

        return trajectory


    def position_at_time(self, t):
        if t <= self.delay:
            s0, v0, a0 = self.start_state
            return s0 + v0 * t + 0.5 * a0 * t * t

        elif t < self.duration + self.delay:
            return self.polynomial(t - self.delay)

        else:
            s1, v1, a1 = self.end_state
            dt = t - self.delay - self.duration
            return s1 + v1 * dt + 0.5 * a1 * dt * dt


    def velocity_at_time(self,t):
        if t <= self.delay:
            v0, a0 = self.start_state[1:]
            return v0 + a0 * t

        elif t < self.duration + self.delay:
            return self.polynomial.deriv(1)(t - self.delay)

        else:
            v1, a1 = self.end_state[1:]
            dt = t - self.delay - self.duration
            return v1 + a1 * dt


    def acceleration_at_time(self, t):
        if t <= self.delay:
            return self.start_state[2]

        elif t < self.duration + self.delay:
            return self.polynomial.deriv(2)(t - self.delay)

        else:
            return self.end_state[2]


    def jerk_at_time(self, t):
        if t <= self.delay or t >= self.delay + self.duration:
            return 0.0

        else:
            return self.polynomial.deriv(3)(t - self.delay)


    def state_at_time(self, t):
        return [self.position_at_time(t), self.velocity_at_time(t), self.acceleration_at_time(t)]


    def time_for_position(self, x, initial_t = None, gamma=0.001, precision=1e-3, max_iterations=1000):
        s1,v1,a1 = self.end_state
        if x > s1 and v1 == 0.0 and a1 == 0.0:
            return float('inf')

        # Derivative of the error term:
        def df(t):
            return -2.0 * (x - self.position_at_time(t)) * self.velocity_at_time(t)

        # Initial guess for gradient descent:
        def calc_initial_t(x):
            t = 0.0
            min_error = 1e9
            min_t = 0.0

            while t < self.total_duration:
                s = self.position_at_time(t)
                error = pow(x - s, 2)
                if error < min_error:
                    min_error = error
                    min_t = t
                t += self.total_duration / 100
            return min_t

        current_t = calc_initial_t(x) if initial_t is None else initial_t
        previous_step_size = float('inf')
        i = 0

        # Perform gradient descent:
        while previous_step_size > precision:
            previous_t = current_t
            current_t += -gamma * df(previous_t)
            previous_step_size = abs(current_t - previous_t)
            if i >= max_iterations:
                break
            i += 1

        return current_t


    def velocity_at_position(self, x):
        if x >= self.end_state[0]:
            return self.end_state[1]

        t = self.time_for_position(x)

        if t == float('inf'):
            return 0.0
        else:
            return self.velocity_at_time(t)


    def state_at_position(self, x):
        state = np.zeros(3)
        state[0] = x

        if x >= self.end_state[0]:
            state[1:3] = self.end_state[1:3]
            return state

        t = self.time_for_position(x)

        if t == float('inf'):
            state[1:3] = [0.0, 0.0]
        else:
            state[1] = self.velocity_at_time(t)
            state[2] = self.acceleration_at_time(t)

        return state



    def cost(self):
        t = np.arange(0.0, self.total_duration, 0.1)
        v = np.vectorize(self.velocity_at_time)(t)
        a = np.vectorize(self.acceleration_at_time)(t)
        j = np.vectorize(self.jerk_at_time)(t)

        if np.any(v < 0.0)   \
          or np.any(a >= self.max_acceleration) \
          or np.any(a <= -self.max_deceleration) \
          or np.any(j >= self.max_jerk) \
          or np.any(j <= -2.0 * self.max_jerk):
            return float('inf')

        return np.sum(a * a + j * j)


    @staticmethod
    def calc_quintic_polynomial(start_state, end_state, duration):
        s0, sd0, sdd0 = start_state
        s1, sd1, sdd1 = end_state

        dT1 = duration
        dT2 = dT1 * dT1
        dT3 = dT2 * dT1
        dT4 = dT3 * dT1
        dT5 = dT4 * dT1

        x = np.array([
            [s1 - s0 - sd0 * dT1 - 0.5 * sdd0 * dT2],
            [sd1 - sd0 - sdd0 * dT1],
            [sdd1 - sdd0]
        ])

        A = np.array([
            [  dT3,    dT4,    dT5],
            [3*dT2,  4*dT3,  5*dT4],
            [6*dT1, 12*dT2, 20*dT3]
        ])

        coeffs1 = np.array([s0, sd0, 0.5 * sdd0])
        coeffs2 = np.dot(np.linalg.inv(A),x)

        return Polynomial(np.concatenate((coeffs1, coeffs2.reshape((3,))), axis=0))


    @staticmethod
    def calc_quartic_polynomial(start_state, end_state, duration):
        s0, sd0, sdd0 = start_state
        s1, sd1, sdd1 = end_state

        dT1 = duration
        dT2 = dT1 * dT1
        dT3 = dT2 * dT1

        x = np.array([
            [sd1 - sd0 - sdd0 * dT1],
            [sdd1 - sdd0]
        ])

        A = np.array([
            [3*dT2,  4*dT3],
            [6*dT1, 12*dT2]
        ])

        coeffs1 = np.array([s0, sd0, 0.5 * sdd0])
        coeffs2 = np.dot(np.linalg.inv(A),x)

        return Polynomial(np.concatenate((coeffs1, coeffs2.reshape((2,))), axis=0))


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    delta_s = 50.0
    v0 = 10.0
    s0 = [0, v0, 0]
    s1 = [delta_s, 0, 0]
    total_duration = 10.0

    trajectories = []

    for duration in np.arange(1.0, total_duration + 0.1, 1.0):
        for delay in np.arange(0.0, total_duration - duration + 0.1, 1.0):
            tr = Trajectory.StoppingTrajectory(s0, s1, duration, delay)

            if tr.cost() < float('inf'):
                trajectories.append(tr)

    fig = plt.figure()
    ax1 = fig.add_subplot(4,1,1)
    ax1.set_title("position")
    ax2 = fig.add_subplot(4,1,2)
    ax2.set_title("velocity")
    ax3 = fig.add_subplot(4,1,3)
    ax3.set_yscale('linear')
    ax3.set_title("accleration")
    ax4 = fig.add_subplot(4,1,4)
    ax4.set_title("jerk")

    for tr in trajectories:
        t = np.arange(0, total_duration, 0.1)
        s = np.vectorize(tr.position_at_time)
        v = np.vectorize(tr.velocity_at_time)
        acc = np.vectorize(tr.acceleration_at_time)
        j = np.vectorize(tr.jerk_at_time)

        ax1.plot(t, s(t))
        ax2.plot(t, v(t))
        ax3.plot(t, acc(t))
        ax4.plot(t, j(t))


    plt.show()


    # print tr.coeffs
    # print
    # print "Testing time_for_position:"
    # for i in range(0, 11):
    #     t = tr.time_for_position(i)
    #     print i, t
    # print
    # print "Testing velocity_at_position:"
    # for x in range(0,21):
    #     v = tr.velocity_at_position(x)
    #     print x, v
