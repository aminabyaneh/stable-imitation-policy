#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import time
import math
import numpy as np

import pybullet as pb
import pybullet_data
import threading as th
import matplotlib.pyplot as plt

from enum import Enum
from typing import Dict, List, Tuple

# in case of later need for recording simulated motions
save_data: bool = False


class ControlModes(Enum):
    """Pybullet control modes, only for the end effector. We use IK to translate
    EF commands to joint vellocities using a PID controller.
    """

    END_EFFECTOR_POSE = pb.POSITION_CONTROL
    END_EFFECTOR_TWIST = pb.VELOCITY_CONTROL


def sample_run():
    ''' After importing the PyBullet module, the first thing to do
    is 'connecting' to the physics simulation. PyBullet is designed
    around a client-server driven API, with a client sending commands
    and a physics server returning the status.

    The DIRECT connection sends the commands directly to the physics engine,
    without using any transport layer and no graphics visualization window,
    and directly returns the status after executing the command. The GUI
    connection will create a new graphical user interface. '''
    physicsClient = pb.connect(pb.GUI)
    pb.setAdditionalSearchPath(pybullet_data.getDataPath())

    ''' By default, there is no gravitational force enabled. setGravity
    lets you set the default gravity force for all objects. '''
    pb.setGravity(0,0,-10)

    ''' The loadURDF will send a command to the physics server to load a
    physics model from a Universal Robot Description File (URDF).'''
    planeId = pb.loadURDF("plane.urdf")
    cubeStartPos = [0, 0, 1]
    cubeStartOrientation = pb.getQuaternionFromEuler([0,0,0])

    boxId = pb.loadURDF("r2d2.urdf", cubeStartPos, cubeStartOrientation)

    sim_time = 1000
    for i in range(sim_time):
        ''' The stepSimulation will perform all the actions in a single forward
        dynamics simulation step such as collision detection, constraint
        solving and integration. '''
        pb.stepSimulation()
        time.sleep(1./240.)

    cubePos, cubeOrn = pb.getBasePositionAndOrientation(boxId)
    print(cubePos,cubeOrn)

    ''' Saving the world allows for later automatic reload. '''
    pb.saveWord(fileName="test")

    ''' A specific or all physics clients can be terminated using disconnect. '''
    pb.disconnect(physicsClient)


class PyBulletExperiments:
    """Handler for pybullet experiments.
    """

    def __init__(self, home: bool = True, start_sim: bool = True, gui: bool = True,
             robot_urdf_path: str = "urdf/gen3_lite.urdf",
             gravity: float = 9.8):
        """Function to initialize a simulation with kinova in pybullet.

        Note: function adapted from pybullet examples.
        Note: Make sure to give the correct path to Gen3Lite URDF.
        """

        if gui:
            # connect to gui and physics engine
            self.__physic_client = pb.connect(pb.GUI)

        # load a plane environment
        pb.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.__plane_id = pb.loadURDF("plane.urdf", [0, 0, -0.3], useFixedBase=True)
        self.__box_id = pb.loadURDF("table/table.urdf", [0, 0, 0], useFixedBase=True)

        # initialize kuka urdf and initialize position
        self.__kinova_id = pb.loadURDF(robot_urdf_path, [0, 0, 0.7], useFixedBase=True)
        pb.resetBasePositionAndOrientation(self.__kinova_id, [0, 0, 0.7], [0, 0, 0, 1])

        # end-effector is the sixth joint
        self.__ef_idx = 6
        self.__n_joints = pb.getNumJoints(self.__kinova_id) - 5 # -5 for the gripper
        print(f'Found {self.__n_joints} active joints for the robot.')

        # null space limits and range
        self.__lower_limits: List = [-.967, -2, -2.96, 0.19, -2.96, -2.09, -3.05]
        self.__upper_limits: List = [.967, 2, 2.96, 2.29, 2.96, 2.09, 3.05]
        self.__joint_ranges: List = [5.8, 4, 5.8, 4, 5.8, 4, 6]
        self.__rest_poses: List = [0, 0, 0, 0.5 * math.pi, 0, -math.pi * 0.5 * 0.66, 0]

        # reset joint states to rest poses
        for i in range(self.__n_joints):
            pb.resetJointState(self.__kinova_id, i, self.__rest_poses[i])

        # set the gravity and time ref
        pb.setGravity(0, 0, -gravity)
        self.__sim_time: float = 0.0
        self.__start_sim = start_sim
        self.__time_step = 1e-2

        # position variables
        self.__prev_pose: List = [0, 0, 0]
        self.__has_prev_pose: bool = 0

        # debug lines will be removed after trail duration
        pb.setRealTimeSimulation(False)
        self.__trail_duration = 20

        # initial pose, end effector points down
        self.__ef_pose = [0.3, -0.2, 0.9] + list(pb.getQuaternionFromEuler([0, -math.pi, 0]))
        self.__ef_vels = [0, 0, 0, 0, 0, 0]
        self.__control_mode = ControlModes.END_EFFECTOR_POSE

        # run the simulation thread
        self.__sim_p = th.Thread(target=self._simulate)
        self.__sim_p.start()

        # send home if requested
        if home: self.home()

    def _simulate(self):
        """Simulation thread. Steps the simulation and moves the robot to target
        positions

        Args:
            control_mode (ControlModes, optional): Switch the control mode. Defaults to
            ControlModes.END_EFFECTOR_POSE, for position control.
        """

        # flag variable for starting the simulation
        while not self.__start_sim:
            time.sleep(0.01)
            continue

        while True:
            # step simulation
            self.__sim_time += self.__time_step
            pb.stepSimulation()

            joint_poses = pb.calculateInverseKinematics(self.__kinova_id, self.__ef_idx,
                                                        self.__ef_pose[:3], self.__ef_pose[3:],
                                                        lowerLimits=self.__lower_limits,
                                                        upperLimits=self.__upper_limits,
                                                        jointRanges=self.__joint_ranges,
                                                        restPoses=self.__rest_poses)

            for i in range(self.__n_joints):
                pb.setJointMotorControl2(bodyIndex=self.__kinova_id,
                                        jointIndex=i,
                                        controlMode=pb.POSITION_CONTROL,
                                        targetPosition=joint_poses[i],
                                        targetVelocity=0,
                                        force=500,
                                        positionGain=0.05,
                                        velocityGain=1)

            ls = pb.getLinkState(self.__kinova_id, self.__ef_idx)
            pose = ls[4]
            if self.__has_prev_pose:
                pb.addUserDebugLine(self.__prev_pose, pose, [1, 0, 0], 1,
                                    lifeTime=self.__trail_duration)

            self.__prev_pose = pose
            self.__has_prev_pose = 1

            if not self.__start_sim:
                break

    def get_endeffector_feedback(self):
        """Return a dict with joints pose and twists feedback.

        Returns:
            Dict: full feedback
        """
        return {"position": pb.getLinkState(self.__kinova_id, self.__n_joints)[4],
                "orientation": pb.getLinkState(self.__kinova_id, self.__n_joints)[5]}

    def toggle_simulation(self, activate: bool):
        """ Enable or disable the simulation flag.

        Args:
            activate (bool): Choose true to start the simulation, false to end it.
        """

        print(f'Toggling the simulation {"ON" if activate else "OFF"}')
        self.__start_sim = activate

    def move(self, data: Dict[str, float]):
        """ Move to a pose or with a velocity depending on the control mode.

        Args:
            data (Dict[str, float]): Pose or twist data. In a dictionary format with keys
            as "linear_x/y/z" and "angular_x/y/z".
        """

        print(f'Moving to {list(data.values())}')
        if self.__control_mode == ControlModes.END_EFFECTOR_POSE:
            self.__ef_pose = list(data.values())
        elif self.__control_mode == ControlModes.END_EFFECTOR_TWIST:
            self.__ef_vels = list(data.values())
            print(f'Twist control is not implemented yet!')


    def execute_trajectory(self, trajectory, is_joint_space: bool = False, wait_time: float = 0.1):
        """Execute a trajectory.

        Args:
            trajectory (List): A list of all the waypoints in the trajectory.
            is_joint_space (bool, optional): False means it's a cartesian trajectory.
                Defaults to False.
            wait_time(float, optional): Time until moving to the next point.
        """

        for waypoint in trajectory:
            self.move(waypoint)
            self.pause(wait_time)

        print('Trajectory execution completed')


    def home(self):
        """ Move to a predefined home position.
        """

        if not self.__start_sim:
            print(f'Simulationi is not active yet!')
            return

        self.__ef_pose = [0.2, 0.2, 0.9] + list(pb.getQuaternionFromEuler([0, -math.pi, 0]))

    def pause(self, secs=2):
        """ Pause for a determined time. Only a wrapper for sleep at this point.

        Args:
            secs (int, optional): Seconds to pause. Defaults to 2.
        """
        time.sleep(secs)

    def grip(self, press: float = 0.8):
        """ Activate the robotic gripper.

        TODO: This function is deactivated for pybullet for now.

        Args:
            press (float, optional): Use the press argument to adjust the
                pressur with fingers. Defaults to 0.8.
        """

        pass

    def __enter__(self):
        """Enter method for with clauses.
        """

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """ Exit method for with clauses.
        """
        try:
            self.__sim_p.join()
            pb.disconnect(self.__physic_client)
            print('Closing pybullet simulator')
        except KeyboardInterrupt:
            print('Closing pybullet simulator')
            self.toggle_simulation(False)
            self.__sim_p.join()
            pb.disconnect(self.__physic_client)


def baseline_sine_motion():
    # TODO: Remove the quaternion and pass euler directly
    start_point = {'linear_x': 0.117,
                    'linear_y': 0.00,
                    'linear_z': 0.955,
                    'angular_x': pb.getQuaternionFromEuler([0, -math.pi, 0])[0],
                    'angular_y': pb.getQuaternionFromEuler([0, -math.pi, 0])[1],
                    'angular_z': pb.getQuaternionFromEuler([0, -math.pi, 0])[2],
                    'angular_w': pb.getQuaternionFromEuler([0, -math.pi, 0])[3]}

    with PyBulletExperiments(home=False) as pbe:
        pbe.move(start_point)
        pbe.pause(30)

        xb, yb, zb = start_point['linear_x'], start_point['linear_y'], start_point['linear_z']

        # generate a trajectory
        task_space_trajectory: List[Dict] = []
        x = xb
        while x < 0.430:
            # calculate new end-effector positions
            x_n = x + 0.005
            y_n = yb + 0.15 * math.sin(50 * (x_n - xb))

            # move to calculated position
            target = {'linear_x': x_n,
                      'linear_y': y_n,
                      'linear_z': zb,
                      'angular_x': pb.getQuaternionFromEuler([0, -math.pi, 0])[0],
                      'angular_y': pb.getQuaternionFromEuler([0, -math.pi, 0])[1],
                      'angular_z': pb.getQuaternionFromEuler([0, -math.pi, 0])[2],
                      'angular_w': pb.getQuaternionFromEuler([0, -math.pi, 0])[3]}

            task_space_trajectory.append(target)

            # feedback simulation
            x = x_n

        print(f'Executing trajectory with {len(task_space_trajectory)} waypoints')
        pbe.execute_trajectory(task_space_trajectory)


def main():
    """ Main entry point and argument parser for the exp file.
    """

    parser = argparse.ArgumentParser(description='Handle basic simulations for learning DS on Kinove Gen3 Lite 6-DOF arm. Note that with minor changes, this code can be re-used for other robotic arms as well.')

    parser.add_argument('-ho', '--home', action='store_true', default=False,
                        help='Send the robot to home pose.')
    parser.add_argument('-hw', '--hand-writing', action='store_true', default=False,
                        help='Use handwriting data for imitation.')
    parser.add_argument('-ms', '--motion-shape', type=str, default='Sine',
                        help='Shape of the trajectory (valid when -hw is enabled).')
    parser.add_argument('-sd', '--save-data', action='store_true', default=False,
                        help='Save the data in the dems folder.')

    args = parser.parse_args()


    global save_data
    save_data = args.save_data


if __name__ == '__main__':
    baseline_sine_motion()