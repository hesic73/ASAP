#!/bin/bash

GPU_ID=$(bash scripts/bash/find_gpu.sh)

echo "Using GPU ID: ${GPU_ID}"

DEVICE="cuda:${GPU_ID}"

HYDRA_FULL_ERROR=1 python humanoidverse/train_agent.py \
    +device=${DEVICE} \
    +exp=motion_tracking \
    +rewards=motion_tracking/reward_motion_tracking_dm_2real \
    num_envs=4096 \
    project_name=MotionTracking \
    experiment_name=MotionTracking_CR7 \
    robot.motion.motion_file="humanoidverse/data/motions/g1_29dof_anneal_23dof/TairanTestbed/singles/0-TairanTestbed_TairanTestbed_CR7_video_CR7_level1_filter_amass.pkl" \
    rewards.reward_penalty_curriculum=True \
    rewards.reward_penalty_degree=0.00001 \
    env.config.resample_motion_when_training=False \
    env.config.termination.terminate_when_motion_far=True \
    env.config.termination_curriculum.terminate_when_motion_far_curriculum=True \
    env.config.termination_curriculum.terminate_when_motion_far_threshold_min=0.3 \
    env.config.termination_curriculum.terminate_when_motion_far_curriculum_degree=0.000025 \
    robot.asset.self_collisions=0
