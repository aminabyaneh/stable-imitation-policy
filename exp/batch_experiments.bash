#!/bin/bash

python=/isaac-sim/python.sh

# SNDS Experiments
python nnds_training.py --neural-tool snds --motion-shape Snake --num-epochs 1000 --eps 0.01 --alpha 0.1
python nnds_training.py --neural-tool snds --motion-shape Snake --num-epochs 1000 --eps 0.1 --alpha 0.01
python nnds_training.py --neural-tool snds --motion-shape Snake --num-epochs 1000 --eps 0.001 --alpha 0.001
python nnds_training.py --neural-tool snds --motion-shape Snake --num-epochs 1000 --eps 0.1 --alpha 0.1

python nnds_training.py --neural-tool snds --motion-shape Snake --num-epochs 1000 --eps 0.01 --alpha 0.1
python nnds_training.py --neural-tool snds --motion-shape Snake --num-epochs 1000 --eps 0.1 --alpha 0.01
python nnds_training.py --neural-tool snds --motion-shape Snake --num-epochs 1000 --eps 0.001 --alpha 0.001
python nnds_training.py --neural-tool snds --motion-shape Snake --num-epochs 1000 --eps 0.1 --alpha 0.1


python nnds_training.py --neural-tool snds --motion-shape Multi_Models_1 --num-epochs 500 --gpu&
python nnds_training.py --neural-tool snds --motion-shape Multi_Models_2 --num-epochs 500 --gpu&
python nnds_training.py --neural-tool snds --motion-shape Multi_Models_3 --num-epochs 500 --gpu&
python nnds_training.py --neural-tool snds --motion-shape Multi_Models_4 --num-epochs 500 --gpu

python nnds_training.py --neural-tool snds --motion-shape Multi_Models_1 --num-epochs 5000 --gpu&
python nnds_training.py --neural-tool snds --motion-shape Multi_Models_2 --num-epochs 5000 --gpu&
python nnds_training.py --neural-tool snds --motion-shape Multi_Models_3 --num-epochs 5000 --gpu&
python nnds_training.py --neural-tool snds --motion-shape Multi_Models_4 --num-epochs 5000 --gpu&

# SDS-EF Experiments
python nnds_training.py --neural-tool sdsef --motion-shape Multi_Models_1 --num-epochs 500
python nnds_training.py --neural-tool sdsef --motion-shape Multi_Models_2 --num-epochs 500
python nnds_training.py --neural-tool sdsef --motion-shape Multi_Models_3 --num-epochs 500
python nnds_training.py --neural-tool sdsef --motion-shape Multi_Models_4 --num-epochs 500

# Vanilla BC
python nnds_training.py --neural-tool nn --motion-shape Multi_Models_1 --num-epochs 50 --gpu&
python nnds_training.py --neural-tool nn --motion-shape Multi_Models_2 --num-epochs 50 --gpu&
python nnds_training.py --neural-tool nn --motion-shape Multi_Models_3 --num-epochs 50 --gpu&
python nnds_training.py --neural-tool nn --motion-shape Multi_Models_4 --num-epochs 50 --gpu&