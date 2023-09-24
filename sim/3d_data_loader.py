#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys, os
import math


from pathlib import Path
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.pardir, 'src'))

from learn_gmm_ds import SE_DS
from utils.utils import calibrate
from utils.plot_tools import plot_trajectory

import numpy as np
from typing import List, Dict


class MathMotionDataset:
    """ Generating real-world and synthetic motions inspired by complex math
    functions.
    """

    def __init__(self, start: np.ndarray = np.array([0.117, 0.00, 0.00]),
            end: np.ndarray = np.array([0.430, 0.00, 0.00]), plot: bool = False):

        self.__start = start
        self.__end = end
        self.__plot = plot
        self.__function = None
        self.__positions: np.ndarray = None
        self.__velocities: np.ndarray = None

    def pick_and_place(self, x, xb, noise_level, period: float = (2 * math.pi) / 50,
            amplitude: float = 0.15):
        out = x + self.prolonged_sine((x - 14), xb, noise_level=0.001, period=(2*np.pi)/15, amplitude=0.1)
        return out

    def prolonged_sine(self, x, xb, noise_level, period: float = (2 * math.pi) / 50,
            amplitude: float = 0.15):
        out = amplitude * np.sin(((2 * np.pi) / period) * (x - xb))
        out += np.random.uniform(-noise_level, noise_level, size=x.shape)
        return out

    def root_parabola(self, x, xb, noise_level, shift: float = 0.16,
            amplitude: float = 0.1):
        out = amplitude * np.sqrt(np.abs(3 - ((20*(x - xb - shift)) ** 2)))
        out += np.random.uniform(-noise_level, noise_level, size=x.shape)
        return out

    def create_expert_dataset(self, n_dems: int = 5, n_samples: int = 1000,
            function: str = "Prolonged_Sine", noise_level: float = 0.003,
            save_dir: Path = ".", calibrated: bool = True,
            normalized: bool = False, scale_factor: float = 150.0):

        assert function.lower() in dir(self), \
            f'Motion not implemented, choose from:\n' \
            f'{[method for method in dir(self) if not method.startswith("_")]}'
        self.__function = function

        start = self.__start
        end = self.__end

        x_b, y_b, z_b = start[0], start[1], start[2]
        x_e, y_e, z_e = end[0], end[1], end[2]

        zs = np.array([z_b] * n_samples)
        zs_dot = np.zeros(shape=zs.shape)

        positions: List[np.ndarray] = []
        velocities: List[np.ndarray] = []

        xs = np.linspace(x_b, x_e, num=n_samples)
        xs_dot = np.diff(xs)
        xs_dot = np.insert(xs_dot, len(xs_dot), 0.00)

        motion_func = getattr(self, function.lower())
        for _ in range(n_dems):
            xs += np.random.uniform(-0.008, 0.008, size=xs.shape)
            # calculate new end-effector positions
            ys = motion_func(xs, x_b, noise_level=noise_level)

            ys_dot = np.diff(ys)
            ys_dot = np.insert(ys_dot, len(ys_dot), 0.00)

            # aggregate positions and velocities
            pos = np.vstack([xs, ys, zs]).T
            pos = calibrate(pos.T).T if calibrated else pos
            positions.append(pos * scale_factor)

            vel = np.vstack([xs_dot, ys_dot, zs_dot]).T
            vel = vel / np.linalg.norm(vel) if normalized else vel
            velocities.append(vel)

        positions = np.vstack(positions)
        velocities = np.vstack(velocities)
        self.__positions, self.__velocities = positions, velocities

        if self.__plot:
            fig = plt.figure(figsize=(12, 8), dpi=120)
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(self.__positions[:, 0], self.__positions[:, 1], self.__positions[:, 2],
                       s=1, c='blue')
            ax.plot(self.__positions[:1, 0], self.__positions[:1, 1], self.__positions[0, 2], c='blue', label=f'{function.replace("_", " ")} Demonstrations')

            ax.set_xlabel('X1', fontsize=16)
            ax.set_ylabel('X2', fontsize=16)
            ax.set_zlabel('X3', fontsize=16)
            ax.legend(fontsize=16, loc='upper center')

            # Show the plot
            plt.savefig(function, dpi=120, bbox_inches='tight')
            plt.show()

        return positions, velocities

    def get_dataset(self):
        """Return the last generated dataset.

        Returns:
            np.ndarray, np.ndarray: Position and velocities.
        """
        return self.__positions, self.__velocities

    def save_dataset(self, save_dir: Path = os.getcwd()):
        """Save the generated data.
        """
        os.makedirs(save_dir, exist_ok=True)

        if self.__positions is not None:
            np.savetxt(f'position_{self.__function.lower()}.csv', self.__positions, delimiter=",")
            np.savetxt(f'velocity_{self.__function.lower()}.csv', self.__velocities, delimiter=",")

# def plot_baseline_policy():
#     xb, yb, zb = start_point[0], start_point[1], start_point[2]
#     dataset_rollouts.append(np.array([xb, yb]) * 150)

#     x = xb
#     for _ in range(20):
#         while x < 0.430:
#             # calculate new end-effector positions
#             x_n = x + 0.001
#             y_n = yb + 0.15 * math.sin(50 * (x_n - xb))

#             # move to calculated position
#             dataset_rollouts.append(np.array([x_n, y_n]) * 150)
#             x = x_n
#     dataset_rollouts = calibrate(np.array(dataset_rollouts).T).T

#     trajectory: List[np.ndarray] = []
#     velocity: List[np.ndarray] = []

#     xb, yb, zb = start_point[0], start_point[1], start_point[2]
#     trajectory.append(np.array([xb, yb]))

#     x = xb
#     y = yb
#     while x < 0.430:
#         # calculate new end-effector positions
#         x_n = x + 0.0003
#         y_n = yb + 0.15 * math.sin(50 * (x_n - xb))

#         # move to calculated position
#         trajectory.append(np.array([x_n, y_n]))

#         diff = np.array([x_n - x, y_n - y])
#         velocity.append(diff / np.linalg.norm(diff))

#         x = x_n
#         y = y_n
#     velocity.append(np.zeros(shape=(2)))

#     trajectory = calibrate(np.array(trajectory).T).T * 150
#     velocity = np.array(velocity)

#     seds = LPV_DS()
#     seds.fit(trajectory, velocity)
#     seds.save('model', ".")

#     # Create a 3D plot
#     fig = plt.figure(figsize=(12, 8), dpi=120)
#     ax = fig.add_subplot(111, projection='3d')
#     ax.scatter(dataset[:, 0], dataset[:, 1], zb, s=1, c='blue')
#     ax.plot(dataset[:1, 0], dataset[:1, 1], zb, c='blue', label='Expert Demonstrations')
#     ax.plot3D(dataset_rollouts[:, 0], dataset_rollouts[:, 1], zb, c='red',
#               label='Policy Action (On Trajectories)')
#     ax.plot(dataset[:1, 0], dataset[:1, 1], zb, c='green', label='Policy Action (Unknown Regions)')

#     x = np.linspace(-50, 10, 10)
#     y = np.linspace(-40, 40, 10)
#     X, Y = np.meshgrid(x, y)

#     for i in range(10):
#         for j in range(10):
#             position = np.array([X[i, j], Y[i, j]])
#             scale = 0.5
#             vel = seds.predict(np.array([position]))[0]
#             vel = vel / np.linalg.norm(vel)

#             ax.quiver(
#                 position[0], position[1], zb,
#                 vel[0] / scale, vel[1] / scale, 0,
#                 color='green',
#                 length=4.5, normalize=True,
#                 lw =1.5,
#             )

#     # Set labels and title
#     ax.set_zlim(-5, 60)
#     ax.set_xlabel('X1', fontsize=16)
#     ax.set_ylabel('X2', fontsize=16)
#     ax.set_zlabel('X3', fontsize=16)
#     ax.legend(fontsize=16, loc='upper center')

#     # Show the plot
#     plt.savefig('3d_illustration', dpi=120, bbox_inches='tight')

#     plt.show()


motion_data = MathMotionDataset(plot=True)
motion_data.create_expert_dataset(normalized=True, function="Root_Parabola", n_dems=50)
motion_data.save_dataset()