#!/bin/bash

python=/isaac-sim/python.sh


# $python nnds_training.py -nt snds -ms Worm -sp -nd 5 -ne 2000 -sd normal_nets&
# $python nnds_training.py -nt snds -ms Sine -sp -nd 5 -ne 2000 -sd normal_nets&
# $python nnds_training.py -nt snds -ms Worm -sp -nd 5 -ne 2000 -sd tiny_nets&
# $python nnds_training.py -nt snds -ms Worm -sp -nd 5 -ne 2000 -sd tiny_nets&
$python nnds_training.py -nt snds -ms Worm -sp -nd 5 -ne 2000 -sd epsilon0.01&
$python nnds_training.py -nt snds -ms Worm -sp -nd 5 -ne 2000 -sd epsilon0.01&
