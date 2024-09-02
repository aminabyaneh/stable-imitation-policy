import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from os import PathLike
from typing import Dict, List


class PlotConfigs:
    """Hardcoded plot configurations.
    """

    COLORS = ["blue", "orange", "green", "purple", "brown"]
    FMTS = ['d--', 'o-', 's:', 'x-.', '*-', 'd--', 'o-']

    FIGURE_SIZE = (10, 10)
    FIGURE_DPI = 120
    POLICY_COLOR = 'grey'
    TRAJECTORY_COLOR = '#377eb8'
    ROLLOUT_ORIGINAL_COLOR = '#ff7f00'
    ROLLOUT_NOISY_COLOR = 'grey'
    ANNOTATE_COLOR = 'black'
    ANNOTATE_SIZE = 40
    TICKS_SIZE = 16
    LABEL_SIZE = 18
    LEGEND_SIZE = 25
    TITLE_SIZE = 18
    FILE_TYPE = "png"
    REFERENCE_SIZE = 18
    ROLLOUT_LINEWIDTH = 0.2


def plot_trajectories(ds, reference: np.ndarray, space_stretch: float = 0.1,
                      n_samples: int = 1000, file_name: str = "test", save_dir: str = "",
                      n_rollouts: int = 3, rollouts_ic_std: int = 0.1,
                      show_legends: bool = False, save_rollouts: bool = True):
    """ Plot a policy for given a DS model and trajectories.

    Args:
        ds (PlanningPolicyInterface): A dynamical system for motion generation task.
        trajectory (np.ndarray): Input trajectory array (n_samples, dim).
        space_stretch (float, optional): How much of the entire space to show in vector map.
            Defaults to 1.

        file_name(str, optional): Name of the plot file. Defaults to "".
        save_dir(str, optional): Provide a save directory for the figure. Leave empty to
            skip saving. Defaults to "".
        n_samples (int, optional): Number of samples in each demonstration. Defaults to 1000.
        n_rollouts (int, optional): Number of trajectories to reproduce. Defaults to 10.
        show_legends (bool, optional): Opt to show the legends. Defaults to True.
    """

    # find trajectory limits
    x_min, x_max, y_min, y_max = find_limits(reference)

    # calibrate the axis
    plt.figure(figsize=PlotConfigs.FIGURE_SIZE, dpi=PlotConfigs.FIGURE_DPI)

    axes = plt.gca()
    axes.set_xlim([x_min - space_stretch, x_max + space_stretch])
    axes.set_ylim([y_min - space_stretch, y_max + space_stretch])
    axes.grid()

    # initial and goal states
    goal_point = reference[-1]
    initial_states = np.array([reference[idx * n_samples] for idx in range(len(reference) // n_samples)])
    print(f'Initial states: {initial_states.shape}, {initial_states}')

    trimed_trajectory_idx = np.random.choice(a=len(reference),
                                             size=int(0.05 * len(reference)),
                                             replace=False)
    trimed_trajectory = np.array(reference[trimed_trajectory_idx])

    # plot the trimmed trajectory
    plt.scatter(trimed_trajectory[:, 0], trimed_trajectory[:, 1],
                color=PlotConfigs.TRAJECTORY_COLOR, marker='o', zorder=2,
                s=PlotConfigs.REFERENCE_SIZE, label='Expert data')

    # original policy rollouts
    dt: float = 0.01
    limit = np.linalg.norm([(x_max - x_min), (y_max - y_min)]) / 100

    for idx, start in enumerate(initial_states):
        simulated_traj: List[np.ndarray] = []
        simulated_traj.append(np.array([start]).reshape(1, 2))

        distance_to_target = np.linalg.norm(simulated_traj[-1] - goal_point)
        while  distance_to_target > limit  and len(simulated_traj) < 5e3:
            vel = ds.predict(simulated_traj[-1])
            simulated_traj.append(simulated_traj[-1] + dt * vel)
            distance_to_target = np.linalg.norm(simulated_traj[-1] - goal_point)

        simulated_traj = np.array(simulated_traj)
        simulated_traj = simulated_traj.reshape(simulated_traj.shape[0],
                                                simulated_traj.shape[2])

        if save_rollouts:
            name = file_name if file_name != "" else 'plot'
            os.makedirs(os.path.join(save_dir, name), exist_ok=True)
            np.save(os.path.join(save_dir, name, f'rollout_original_{idx}'), simulated_traj)

        # plot
        plt.plot(simulated_traj[:, 0], simulated_traj[:, 1], color=PlotConfigs.ROLLOUT_ORIGINAL_COLOR,
                 linewidth=PlotConfigs.ROLLOUT_LINEWIDTH * 10, zorder=1, label='True IC')

    # noisy policy rollouts
    selected_indices = np.random.choice(len(initial_states), size=n_rollouts * 10,
                                        replace=True)
    selected_states = initial_states[selected_indices]

    noise = np.random.uniform(-rollouts_ic_std, +rollouts_ic_std, selected_states.shape)
    noisy_initial_states = selected_states + noise

    for idx, start in enumerate(noisy_initial_states):
        simulated_traj: List[np.ndarray] = []
        simulated_traj.append(np.array([start]).reshape(1, 2))

        distance_to_target = np.linalg.norm(simulated_traj[-1] - goal_point)
        while  distance_to_target > limit  and len(simulated_traj) < 5e3:
            vel = ds.predict(simulated_traj[-1])
            simulated_traj.append(simulated_traj[-1] + dt * vel)
            distance_to_target = np.linalg.norm(simulated_traj[-1] - goal_point)

        simulated_traj = np.array(simulated_traj)
        simulated_traj = simulated_traj.reshape(simulated_traj.shape[0],
                                                simulated_traj.shape[2])

        if save_rollouts:
            name = file_name if file_name != "" else 'plot'
            np.save(os.path.join(save_dir, name, f'rollout_noisy_{idx}'), simulated_traj)
        plt.plot(simulated_traj[:, 0], simulated_traj[:, 1], color=PlotConfigs.ROLLOUT_NOISY_COLOR,
                 linewidth=PlotConfigs.ROLLOUT_LINEWIDTH)

    start_handle = plt.scatter(initial_states[:, 0], initial_states[:, 1], marker='x',
                               color=PlotConfigs.ANNOTATE_COLOR, linewidth=2,
                               s=PlotConfigs.ANNOTATE_SIZE, label='Start', zorder=3)

    start_handle = plt.scatter(noisy_initial_states[:, 0], noisy_initial_states[:, 1], marker='x',
                               color=PlotConfigs.ANNOTATE_COLOR, linewidth=2,
                               s=PlotConfigs.ANNOTATE_SIZE, label='Start', zorder=3)

    target_handle = plt.scatter(reference[-1, 0], reference[-1, 1], marker='*',
                                color=PlotConfigs.ANNOTATE_COLOR, linewidth=2,
                                s=(4 * PlotConfigs.ANNOTATE_SIZE), label='Target', zorder=3)

    plt.gca().set_xticklabels([])
    plt.gca().set_yticklabels([])

    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, name), dpi=PlotConfigs.FIGURE_DPI, bbox_inches='tight')


def find_limits(trajectory):
    """ Find the trajectory limits.

    Args:
        trajectory (np.ndarray): The given trajectory for finding limitations. Can be 2 or
            3 dimensions.

    Raises:
        NotSupportedError: Dimensions more than 3 are invalid.

    Returns:
        Tuple: A tuple of limits based on the dimensions (4 or 6 elements)
    """

    dimension = trajectory.shape[1]
    if dimension == 2:
        x_min = np.min(trajectory[:, 0])
        y_min = np.min(trajectory[:, 1])
        x_max = np.max(trajectory[:, 0])
        y_max = np.max(trajectory[:, 1])
        return x_min, x_max, y_min, y_max

    else:
        raise NotImplementedError('Dimension not supported')
