#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import time
import pickle

import argparse
import numpy as np

from typing import Dict, List, Tuple, Callable
from alive_progress import alive_bar

sys.path.append(os.path.join(os.pardir, 'src'))

from utils.utils import mse
from seds_learning import learn_seds_policy
from plyds_learning import learn_plyds_policy
from rlds_training import train_rl_policy
from nnds_training import train_neural_policy

from utils.log_config import logger
from utils.data_loader import lasa_selected_motions


def batch_learning_stable_policy(method: str, n_trials: int,
    save_dir: str, available_methods: Dict):
    """ Run policy learning methods on all available motions for several random seeds
    and report the results.

    TODO: Save results as DataFrames and csv for easier parsing and access to samples.
    TODO: Add batch indexing instead of timestamps.

    Args:
        n_trials (int): Set the number of trials per each demonstrated motion.

        plyds_deg (int, optional): Maximum degree of the dynamical system. Defaults to 2.
        lpf_deg (int, optional): Maximum complexity of lyapunov potential function. Defaults to 2.
        mode (str, optional): Switch between train and test modes. Defaults to 'train'.
        motion_shape (str, optional): Shape of the motion to load from Lasa dataset. Defaults to "G".
        model_name (str, optional): Name of the model to be saved. Defaults to 'test'.
        simplify_lpf (bool, optional): Switch on to use the most simple quadratic LPF.
        plot (bool, optional): Choose to show plots or not. Defaults to False.
        save (bool, optional): Save the model. Defaults to False.
    """

    ''' Verify the method '''
    assert method in available_methods, "Method not implemented yet, build one based on PlanningPolicyInterface, and wrap it in a experiment function like learn_plyds_policy."
    learn_policy, arguments = available_methods[method]

    trials_mse: List[float] = []
    learning_times: List[float] = []
    results_dict: Dict[str, Dict[str, float]] = {}

    ''' Trials loop '''
    with alive_bar(len(lasa_selected_motions) * n_trials) as bar:
        for motion in lasa_selected_motions:
            ''' Progress bar '''
            bar.title(f'{method} Learning {motion}')
            arguments["motion_shape"] = motion

            ''' Trial loop '''
            for _ in range(n_trials):

                ''' Policy learning '''
                t_start = time.time()
                error, _ = learn_policy(**arguments)
                t_total = time.time() - t_start

                learning_times.append(t_total)
                trials_mse.append(error)
                bar()

            ''' Store results '''
            results_dict[motion] = {
                'mean': np.mean(np.array(trials_mse)),
                'std': np.std(np.array(trials_mse)),
                'time': np.mean(np.array(learning_times))
                }

            trials_mse.clear()
            learning_times.clear()


    ''' Save results '''
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, 'results.pkl'), 'wb') as f:
        pickle.dump(results_dict, f)

    logger.info(f'Batch experiments concluded: \n{results_dict}')



def main():
    """ Main entry point and argument parser for the exp file.
    """

    parser = argparse.ArgumentParser(description='Batch experiments CLI interface.')

    # common arguments
    parser.add_argument('-pl', '--policy-learner', type=str, default="PLYDS",
        help='Name of the trained model.')
    parser.add_argument('-sp', '--show-plots', action='store_true', default=False,
        help='Show plots of final result and trajectories.')
    parser.add_argument('-nt', '--n-trials', type=int, default=10,
        help='Number of trials per demonstrated motion.')
    parser.add_argument('-sm', '--save-models', action='store_true', default=False,
        help='Keep a copy of the models in the res folder.')
    parser.add_argument('-mn', '--model-name', type=str, default="batch",
        help='Name of the trained model for saving.')
    parser.add_argument('-sd', '--save-dir', type=str,
        default=os.path.join(os.pardir, 'res', 'batch_result'),
        help='Name of the trained model for saving.')
    parser.add_argument('-nd', '--num-demonstrations', type=int, default=50,
        help='Number of additional demonstrations to the original dataset.')

    # plyds arguments
    parser.add_argument('-dsd', '--ds-degree', type=int, default=2,
        help='Complexity of the polynomial dynamical system.')
    parser.add_argument('-lpfd', '--lpf-degree', type=int, default=2,
        help='Complexity of the stability Lyapunov function.')
    parser.add_argument('-o', '--optimizer', type=str, default="cvxpy",
        help='Switch between scipy and cvxpy optimizers.')
    parser.add_argument('-st', '--set-tolerance', type=int, default=5,
        help='Number of trials per demonstrated motion only for cvxpy.')
    parser.add_argument('-ls', '--lpf-simplification', action='store_true', default=True,
        help='Use lyapunov simplification if activated.')

    # gail and bc arguments
    parser.add_argument('-pa', '--policy-agent', type=str, default="PPO",
        help='Choose the policy learner agent.')
    parser.add_argument('-neb', '--num-epochs-bc', type=int, default=10,
        help='Number of training epochs for BC.')
    parser.add_argument('-ntg', '--num-timesteps-gail', type=int, default=int(3e4),
        help='Number of training timesteps for GAIL.')
    parser.add_argument('-wp', '--warm-policy', action='store_true', default=True,
        help='Use a BC policy to kick-start GAIL.')

    # parse arguments
    args = parser.parse_args()

    # plyds args
    plyds_args: Dict = dict(plyds_deg=args.ds_degree, lpf_deg=args.lpf_degree,
        optimizer=args.optimizer, simplify_lpf=args.lpf_simplification, plot=args.show_plots,
        save=args.save_models, save_dir=os.path.join(args.save_dir, 'plyds'),
        model_name=args.model_name, n_dems=args.num_demonstrations,
        tol=args.set_tolerance, motion_shape=None)

    # gail and bc args
    rl_args = dict(learning_method=args.policy_learner,
        policy_agent=args.policy_agent, n_dems=args.num_demonstrations,
        n_epochs_bc=args.num_epochs_bc, n_timesteps_gail=args.num_timesteps_gail,
        warm_policy=args.warm_policy, plot=args.show_plots,
        model_name=args.model_name, save=args.save_models,
        save_dir=os.path.join(args.save_dir, 'rlds'), motion_shape=None)

    # seds args
    seds_args = dict(motion_shape=None, show_plots=args.show_plots, save=args.save_models,
                     save_dir=os.path.join(args.save_dir, 'seds'), model_name=args.model_name,
                     n_dems=args.num_demonstrations)
    neural_args = dict(motion_shape=None, show_plots=args.show_plots, save=args.save_models,
                       save_dir=os.path.join(args.save_dir, 'nlds'), model_name=args.model_name)

    # arrange methods and args
    available_methods: Dict[str, Tuple[Callable, Dict]] = {
        "PLYDS": (learn_plyds_policy, plyds_args),
        "SEDS": (learn_seds_policy, seds_args),
        "GAIL": (train_rl_policy, rl_args),
        "BC": (train_rl_policy, rl_args),
        "NN": (train_neural_policy, neural_args),
        "LIPNET": (train_neural_policy, neural_args)
        }

    # launch batch experiments
    batch_learning_stable_policy(args.policy_learner, args.n_trials, args.save_dir,
        available_methods)


if __name__ == '__main__':
    main()
