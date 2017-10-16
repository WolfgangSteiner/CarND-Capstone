from trajectory import Trajectory
import numpy as np

class TrajectoryGenerator(object):
    def __init__(self, start_state, end_state):
        self.start_state = start_state
        self.end_state = end_state
        self.planning_horizon = 10.0
        self.acceeration_weight = 1.0
        self.jerk_weight = 1.0
        self.trajectories = []
        self.time_step = 0.5


    @staticmethod
    def CreateGenerator(start_state, end_state, generator_func, **kwargs):
        gen = TrajectoryGenerator(start_state, end_state)
        for duration in np.arange(1.0, gen.planning_horizon + 0.1, gen.time_step):
            for delay in np.arange(0.0, gen.planning_horizon - duration + 0.1, gen.time_step):
                tr = generator_func(gen.start_state, gen.end_state, duration, delay, **kwargs)
                if tr.cost() < float('inf'):
                    gen.trajectories.append(tr)

        return gen


    @staticmethod
    def StoppingTrajectoryGenerator(start_state, end_state, **kwargs):
        return TrajectoryGenerator.CreateGenerator(start_state, end_state, Trajectory.StoppingTrajectory, **kwargs)


    @staticmethod
    def VelocityKeepingTrajectoryGenerator(start_state, end_state, **kwargs):
        return TrajectoryGenerator.CreateGenerator(start_state, end_state, Trajectory.VelocityKeepingTrajectory, **kwargs)


    def minimum_cost_trajectory(self):
        min_cost = float('inf')
        min_cost_trajectory = None

        for tr in self.trajectories:
            c = tr.cost()
            if c < min_cost:
                min_cost = c
                min_cost_trajectory = tr

        return min_cost_trajectory


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    v0 = 20.0
    delta_s = 50.0
    s0 = [0, v0, 0]
    s1 = [delta_s, 0, 0]
    gen = TrajectoryGenerator.StoppingTrajectoryGenerator(s0, s1)

    fig = plt.figure()
    num_figs = 6
    ax1 = fig.add_subplot(num_figs,1,1)
    ax1.set_title("position")
    ax2 = fig.add_subplot(num_figs,1,2)
    ax2.set_title("velocity")
    ax3 = fig.add_subplot(num_figs,1,3)
    ax3.set_yscale('linear')
    ax3.set_title("accleration")
    ax4 = fig.add_subplot(num_figs,1,4)
    ax4.set_title("jerk")
    ax5 = fig.add_subplot(num_figs,1,5)
    ax5.set_title("time at position")
    ax6 = fig.add_subplot(num_figs,1,6)
    ax6.set_title("velocity at position")

    best_trajectory = gen.minimum_cost_trajectory()
    t = np.arange(0, gen.planning_horizon, 0.1)

    def plot_trajectory(tr, color):
        s = np.vectorize(tr.position_at_time)
        v = np.vectorize(tr.velocity_at_time)
        a = np.vectorize(tr.acceleration_at_time)
        j = np.vectorize(tr.jerk_at_time)
        ax1.plot(t, s(t), color=color)
        ax2.plot(t, v(t), color=color)
        ax3.plot(t, a(t), color=color)
        ax4.plot(t, j(t), color=color)


    for tr in gen.trajectories:
        plot_trajectory(tr, '#888888')
    plot_trajectory(best_trajectory, '#00ff00')

    ws = np.arange(0,delta_s,0.5)
    t_at_ws = np.vectorize(best_trajectory.time_for_position)(ws)
    v_at_ws = np.vectorize(best_trajectory.velocity_at_position)(ws)
    ax5.plot(ws, t_at_ws)
    ax6.plot(ws, v_at_ws)



    plt.show()
