#!/usr/bin/env bash
set -euo pipefail

source "/home/user/miniconda3/etc/profile.d/conda.sh"
conda activate aerialgym
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json

cd "/home/user/aerial_gym_simulator/environment/aerial_gym/rl_training/rl_games"

COMMON_ARGS=(--task=position_setpoint_task --num_envs=32 --headless=True --use_warp=True)

echo "===== START PPO ====="
python runner.py --file=ppo_position_setpoint_dense.yaml "${COMMON_ARGS[@]}"
echo "===== START DDPG ====="
python runner.py --file=../offpolicy/configs/ddpg_position_setpoint_dense.yaml "${COMMON_ARGS[@]}"
echo "===== START DDPG+OURS ====="
python runner.py --file=../offpolicy/configs/ddpg_ours_position_setpoint_dense.yaml "${COMMON_ARGS[@]}"
echo "===== START TD3 ====="
python runner.py --file=../offpolicy/configs/td3_position_setpoint_dense.yaml "${COMMON_ARGS[@]}"
echo "===== START TD3+HER ====="
python runner.py --file=../offpolicy/configs/td3_her_position_setpoint_dense.yaml "${COMMON_ARGS[@]}"
echo "===== START TD3+OURS ====="
python runner.py --file=../offpolicy/configs/td3_ours_position_setpoint_dense.yaml "${COMMON_ARGS[@]}"
echo "===== START SAC ====="
python runner.py --file=../offpolicy/configs/sac_position_setpoint_dense.yaml "${COMMON_ARGS[@]}"
echo "===== START SAC+HER ====="
python runner.py --file=../offpolicy/configs/sac_her_position_setpoint_dense.yaml "${COMMON_ARGS[@]}"
echo "===== START SAC+OURS ====="
python runner.py --file=../offpolicy/configs/sac_ours_position_setpoint_dense.yaml "${COMMON_ARGS[@]}"
echo "===== START TDM ====="
python runner.py --file=../offpolicy/configs/tdm_position_setpoint_dense.yaml "${COMMON_ARGS[@]}"
echo "===== START CRL ====="
python runner.py --file=../offpolicy/configs/crl_position_setpoint_dense.yaml "${COMMON_ARGS[@]}"
