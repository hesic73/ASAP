#!/bin/bash

GPU_ID=$(bash scripts/bash/find_gpu.sh)

echo "Using GPU ID: ${GPU_ID}"

DEVICE="cuda:${GPU_ID}"

xvfb-run -s "-screen 0 800x600x24" python humanoidverse/eval_offline.py \
    +device=${DEVICE} \
    +opt=my_eval_callbacks \
    +checkpoint=/home/sichenghe/25summer/ASAP/logs/MotionTracking/20250714_120154-ref_motion_phase_cr7-motion_tracking-g1_29dof_anneal_23dof/model_0.pt
