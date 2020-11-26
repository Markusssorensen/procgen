# -*- coding: utf-8 -*-
"""
Created on Thu Nov 26 14:21:06 2020

@author: Markus
"""

#!/bin/sh
#BSUB -q gpuv100
#BSUB -gpu "num=1"
#BSUB -J Model2_Procgen
#BSUB -n 1
#BSUB -W 12:00
#BSUB -R "rusage[mem=32GB]"
#BSUB -o logs/%J.out
#BSUB -e logs/%J.err

<loading of modules, dependencies etc.>

module load python3/3.6.2
module load cuda/8.0
module load cudnn/v7.0-prod-cuda8
module load ffmpeg/4.2.2

echo "Running script..."
python3 model2_hpc.py
