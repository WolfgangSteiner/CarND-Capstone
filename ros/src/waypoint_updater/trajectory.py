import numpy as np
from numpy.polynomial import Polynomial

class Trajectory:
    def __init__(self, start_state, end_state, dt):
        self.start_state = start_state
        self.end_state = end_state
        self.dt = dt
        self.polynomial = Trajectory.calc_polynomial(start_state, end_state, dt)


    def position_at_time(self, t):
        return self.polynomial(t)


    def velocity_at_time(self,t):
        return self.polynomial.deriv(1)(t)


    def acceleration_at_time(self, t):
        return self.polynomial.deriv(2)(t)


    def jerk_at_time(self, t):
        return self.polynomial.deriv(3)(t)


    def time_for_position(self, x):
        # Derivative of the error term:
        def df(t):
            return -2.0 * (x - self.position_at_time(t)) * self.velocity_at_time(t)

        # Initial guess for gradient descent:
        def initial_t(x):
            t = 0.0
            min_error = 1e9
            min_t = 0.0

            while t < self.dt:
                s = self.position_at_time(t)
                error = pow(x - s, 2)
                if error < min_error:
                    min_error = error
                    min_t = t
                t += self.dt / 100
            return min_t

        MAX_ITER = 1000
        i = 0
        current_t = initial_t(x)
        gamma = 0.01 # step size multiplier
        precision = 0.00001
        previous_step_size = self.dt

        while previous_step_size > precision:
            previous_t = current_t
            current_t += -gamma * df(previous_t)
            previous_step_size = abs(current_t - previous_t)
            if i >= MAX_ITER:
                break
            i += 1

        return current_t


    def velocity_at_position(self, x):
        t = self.time_for_position(x)
        return self.velocity_at_time(t)


    @staticmethod
    def calc_polynomial(start_state, end_state, dt):
        s0, sd0, sdd0 = start_state
        s1, sd1, sdd1 = end_state

        dt1 = dt
        dt2 = dt * dt
        dt3 = dt2 * dt
        dt4 = dt3 * dt
        dt5 = dt4 * dt

        x = np.array([
            [s1 - s0 - sd0 * dt1 - 0.5 * sdd0 * dt2],
            [sd1 - sd0 - sdd0 * dt1],
            [sdd1 - sdd0]
        ])

        A = np.array([
            [  dt3,    dt4,    dt5],
            [3*dt2,  4*dt3,  5*dt4],
            [6*dt1, 12*dt2, 20*dt3]
        ])

        coeffs1 = np.array([s0, sd0, 0.5 * sdd0])
        coeffs2 = np.dot(np.linalg.inv(A),x)

        return Polynomial(np.concatenate((coeffs1, coeffs2.reshape((3,))), axis=0))


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    delta_s = 20.0
    v0 = 10.0
    s0 = [0, v0, 0]
    s1 = [delta_s, 0, 0]

    dt_min = delta_s / v0
    dt = dt_min * 2.0
    tr = Trajectory(s0, s1, dt)

    t = np.arange(0, dt, 0.01)
    s = np.vectorize(tr.position_at_time)
    v = np.vectorize(tr.velocity_at_time)
    a = np.vectorize(tr.acceleration_at_time)
    j = np.vectorize(tr.jerk_at_time)

    fig = plt.figure()
    ax1 = fig.add_subplot(4,1,1)
    ax1.set_title("position")
    ax1.plot(t, s(t))
    ax2 = fig.add_subplot(4,1,2)
    ax2.set_title("velocity")
    ax2.plot(t, v(t))
    ax3 = fig.add_subplot(4,1,3)
    ax3.set_title("accleration")
    ax3.plot(t, a(t))
    ax4 = fig.add_subplot(4,1,4)
    ax4.set_title("jerk")
    ax4.plot(t, j(t))
    plt.show()

    print tr.coeffs
    print
    print "Testing time_for_position:"
    for i in range(0, 11):
        t = tr.time_for_position(i)
        print i, t
    print
    print "Testing velocity_at_position:"
    for x in range(0,21):
        v = tr.velocity_at_position(x)
        print x, v
