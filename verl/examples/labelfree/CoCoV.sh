#!/bin/bash

# ============================================================
# CoCoV: Consistency-based Cooperative Verification Training
# ============================================================
# Diversity-regularized PPO training with two-stage self-
# verification. Uses pass_grpo_penalized advantage estimator.
#
# Usage:
#   bash CoCoV.sh [OPTIONS] [-- EXTRA_HYDRA_ARGS...]
#
# Options:
#   --task TASK         Training task (default: DAPO)
#                       Appends "-TTT" automatically.
#                       Options: DAPO, AIME, AMC, AIME25, math_train
#   --backbone MODEL    Model name or full path (default: Qwen3-4B-Base)
#                       Plain name => $DEFAULT_MODEL_DIR/<MODEL>
#                       Path with "/" => used directly
#   --clip-high [VAL]   PPO clip ratio high:
#                         omitted => 0.2 | flag only => 0.28 | =VAL
#   --ent COEFF         Entropy regularization coeff (default: 0.000)
#   --temp TEMP         Rollout sampling temperature (default: 1.0)
#   -h, --help          Show this help and exit
#
# Environment Variables (optional):
#   WANDB_ENTITY      - W&B entity/team name
#   DEFAULT_MODEL_DIR - Model weights base dir (default: /your/default/model/dir)
#   OUTPUT_BASE_DIR   - Checkpoint/log base dir (default: ./outputs)
#
# Examples:
#   bash CoCoV.sh
#   bash CoCoV.sh --task AIME --backbone Qwen3-8B-Base
#   bash CoCoV.sh --backbone /path/to/my/model
#   bash CoCoV.sh --clip-high 0.30 --ent 0.003
#   bash CoCoV.sh --temp 0.8 -- trainer.n_gpus_per_node=8
# ============================================================

unset VLLM_ATTENTION_BACKEND
export VLLM_USE_V1=1

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --task)
            TASK="$2"
            shift 2
            ;;
        --backbone)
            BACKBONE="$2"
            shift 2
            ;;
        --clip-high)
            CLIP_HIGH="true"
            CLIP_SPECIFIED="true"
            CLIP_MODE="high"
            if [[ -n "$2" && "$2" != --* ]]; then
              CLIP_VALUE="$2"
              shift 2
            else
              shift 1
            fi
            ;;
        --clip-high=*)
            CLIP_HIGH="true"
            CLIP_SPECIFIED="true"
            CLIP_MODE="high"
            CLIP_VALUE="${1#--clip-high=}"
            shift 1
            ;;
        --ent)
            if [[ -z "$2" || "$2" == --* ]]; then
              shift 1
            else
              ENT="$2"
              shift 2
            fi
            ;;
        --ent=*)
            ENT="${1#--ent=}"
            shift 1
            ;;
        --temp)
            TEMP="$2"
            shift 2
            ;;
        --temp=*)
            TEMP="${1#--temp=}"
            shift 1
            ;;
        -h|--help)
            echo "Usage: $0 [--task TASK] [--backbone MODEL] [--clip-high[=VAL]] [--ent COEFF] [--temp TEMP]"
            echo "  --task      Task name (default: DAPO)"
            echo "  --backbone  Model name or path (default: Qwen3-4B-Base)"
            echo "  --clip-high[=VAL] clip ratio: omitted=0.2; flag=0.28; =VAL"
            echo "  --ent       Entropy coeff (default: 0.000)"
            echo "  --temp      Temperature (default: 1.0)"
            echo "  -h, --help  Show help"
            exit 0
            ;;
        *)
            shift
            ;;
    esac
done

# Set default values
TASK=${TASK:-"DAPO"}
BACKBONE=${BACKBONE:-"Qwen3-4B-Base"}
CLIP_HIGH=${CLIP_HIGH:-"false"}
CLIP_SPECIFIED=${CLIP_SPECIFIED:-"false"}
CLIP_VALUE=${CLIP_VALUE:-""}
CLIP_MODE=${CLIP_MODE:-""}
ENT=${ENT:-"0.000"}
TEMP=${TEMP:-"1.0"}

# Set entropy coefficient (numerical) based on --ent
ENTROPY_COEFF=$ENT
RAW_TASK="$TASK"
if [ "$RAW_TASK" = "math_train" ]; then
  TASK="MATH-TTT"
else
  TASK="$TASK-TTT"
fi

pkill -f "python.*main_ppo" || true
pkill -f "python.*main_dapo" || true
pkill -f "multiprocessing.spawn" || true
pkill -f "test_three_datasets.sh" || true
pkill -f "python.*scripts.model_merger" || true
ray stop --force 2>/dev/null || true
sleep 2
echo "========================="

# ------------------------------------------------------------

DATE=$(date +%m%d)
# TIME_TAG=$(date +%H%M%S)
TIME_TAG=123456

ADVANTAGE="pass_grpo_penalized"

echo "=== Basic Configuration Information ==="
echo "Task: $TASK"
echo "Backbone model: $BACKBONE"
echo "Advantage estimator: $ADVANTAGE"
echo "====================================="

# Set K value
K=4
MAX_PROMPT_LENGTH=1024
MAX_RESPONSE_LENGTH=$((1024 * $K))
MAX_TOKEN_LEN=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))
MAX_TOKEN_LEN2=$((MAX_TOKEN_LEN * 2))
if [ "$K" -gt 13 ]; then
  N=4
else
  N=16
fi
  
# Set EPISODE
EPISODE=1
DATA_TRAIN_BATCH_SIZE=32
N_VOTES_PER_PROMPT=64
N_SAMPLES_PER_PROMPT=32
MINI_BATCH_SIZE=1
MICRO_BATCH_SIZE=4

DATA_LOCAL_DIR="data"
DEFAULT_MODEL_DIR=${DEFAULT_MODEL_DIR:-"/your/default/model/dir"}
# Parse backbone model path and safe name
CHAT_TEMPLATE=""
if [[ "$BACKBONE" == *"/"* ]]; then
  BACKBONE_PATH="$BACKBONE"
  BACKBONE_NAME="${BACKBONE##*/}"
else
  BACKBONE_PATH="${DEFAULT_MODEL_DIR}/${BACKBONE}"
  BACKBONE_NAME="$BACKBONE"
fi

echo "Parsed model path: $BACKBONE_PATH"
echo "Parsed model name: $BACKBONE_NAME"

MODEL="${TASK}-${BACKBONE_NAME}"

EXPERIMENT="TTRL-CoCoV-K${K}-Adv${ADVANTAGE}"

# Set clip_ratio_high value and experiment name suffix
if [ "$CLIP_SPECIFIED" = "true" ]; then
  if [ -n "$CLIP_VALUE" ]; then
    CLIP_RATIO_HIGH=$CLIP_VALUE
  else
    CLIP_RATIO_HIGH=0.28
  fi
  if [ "$CLIP_HIGH" = "true" ]; then
    EXPERIMENT="${EXPERIMENT}-ClipHigh"
  fi
else
  CLIP_RATIO_HIGH=0.2
fi

# if RAW_TASK is math_train, use our preprocessed parquet; else follow original logic
if [ "$RAW_TASK" = "math_train" ]; then
  TRAIN_FILES="math_train_ttrl.parquet"
else
  if [[ "$TASK" == *"AIME"* ]]; then
    TRAIN_FILES="train-simplerl-16.parquet"
  else
    TRAIN_FILES="train-simplerl.parquet"
  fi
fi

# Set WANDB_PROJECT based on TASK
if [ "$RAW_TASK" = "math_train" ]; then
  WANDB_PROJECT="TTRL_MATH_TRAIN"
  EXPERIMENT="${EXPERIMENT}-MATH_TRAIN"
elif [ "$TASK" = "AIME-TTT" ]; then
  WANDB_PROJECT="TTRL-AIME24"
elif [ "$TASK" = "AMC-TTT" ]; then
  WANDB_PROJECT="TTRL-AMC"
else
  WANDB_PROJECT="TTRL-MATH500"
fi


if [ "$CLIP_HIGH" = "true" ]; then
  EXPERIMENT="${EXPERIMENT}-ClipHigh"
fi

# Always include specific entropy coefficient in experiment name
EXPERIMENT="${EXPERIMENT}-Ent${ENTROPY_COEFF}"


LOG_NAME="${EXPERIMENT}-${MODEL}"
OUTPUT_BASE_DIR=${OUTPUT_BASE_DIR:-"./outputs"}
OUTPUT_DIR="${OUTPUT_BASE_DIR}/${WANDB_PROJECT}/${MODEL}/${EXPERIMENT}/${TIME_TAG}"



echo "=== TTRL Training Configuration ==="
echo "Task: $TASK"
echo "Backbone model: $BACKBONE"
echo "Advantage estimator: $ADVANTAGE"
if [[ "$ENTROPY_COEFF" != "0" && "$ENTROPY_COEFF" != "0.0" && "$ENTROPY_COEFF" != "0.00" && "$ENTROPY_COEFF" != "0.000" ]]; then
  ENT_ENABLED="true"
else
  ENT_ENABLED="false"
fi
echo "Enable entropy regularization: $ENT_ENABLED"
echo "Entropy coefficient: $ENTROPY_COEFF"
echo "Output directory: $OUTPUT_DIR"
echo "Experiment name: $LOG_NAME"
echo "==============================="

# ============================================================
# Start PPO Training
# ============================================================
python -m verl.trainer.main_ppo \
  reward_model.reward_manager=ttrl \
  reward_model.reward_kwargs.n_samples_per_prompt=$N_SAMPLES_PER_PROMPT \
  reward_model.reward_kwargs.n_votes_per_prompt=$N_VOTES_PER_PROMPT \
  reward_model.reward_kwargs.mode="train" \
  data.train_files=["$DATA_LOCAL_DIR/$TASK/train-simplerl.parquet"] \
  data.val_files=["$DATA_LOCAL_DIR/AIME-TTT/test-simplerl.parquet","$DATA_LOCAL_DIR/MATH-TTT/test-simplerl.parquet","$DATA_LOCAL_DIR/AMC-TTT/test-simplerl.parquet","$DATA_LOCAL_DIR/AIME25-TTT/test-simplerl.parquet","$DATA_LOCAL_DIR/GPQA-TTT/test-simplerl.parquet"] \
  data.max_prompt_length=$MAX_PROMPT_LENGTH \
  data.max_response_length=$MAX_RESPONSE_LENGTH \
  data.train_batch_size=$DATA_TRAIN_BATCH_SIZE \
  data.filter_overlong_prompts=True \
  data.truncation='error' \
  actor_rollout_ref.model.path=$BACKBONE_PATH \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.actor.ppo_mini_batch_size=$MINI_BATCH_SIZE \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=0.001 \
  actor_rollout_ref.actor.clip_ratio_high=$CLIP_RATIO_HIGH \
  actor_rollout_ref.actor.optim.lr=5e-7 \
  actor_rollout_ref.actor.entropy_coeff=$ENTROPY_COEFF \
  actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.03 \
  actor_rollout_ref.actor.optim.warmup_style='cosine' \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$((MAX_TOKEN_LEN2)) \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.temperature=$TEMP \
  actor_rollout_ref.rollout.enforce_eager=False \
  actor_rollout_ref.rollout.free_cache_engine=False \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
  actor_rollout_ref.rollout.do_vote=True \
  actor_rollout_ref.rollout.n_vote=$N_VOTES_PER_PROMPT \
  actor_rollout_ref.rollout.n=$N_SAMPLES_PER_PROMPT \
  actor_rollout_ref.rollout.val_kwargs.do_sample=False \
  actor_rollout_ref.rollout.val_kwargs.top_p=0 \
  actor_rollout_ref.rollout.val_kwargs.temperature=0 \
  actor_rollout_ref.rollout.max_model_len=$((MAX_TOKEN_LEN)) \
  actor_rollout_ref.rollout.max_num_batched_tokens=$((MAX_TOKEN_LEN2)) \
  critic.optim.lr=9e-6 \
  critic.model.use_remove_padding=True \
  critic.model.path=$BACKBONE_PATH \
  critic.model.enable_gradient_checkpointing=True \
  critic.ppo_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE \
  critic.model.fsdp_config.param_offload=False \
  critic.model.fsdp_config.optimizer_offload=False \
  algorithm.kl_ctrl.kl_coef=0.00 \
  algorithm.k=4 \
  algorithm.adv_estimator=$ADVANTAGE \
  two_stage_verify=True \
  two_stage_mode='sampling' \
  two_stage_n=8 \
  two_stage_micro_batch_size=16 \
  two_stage_max_new_tokens=2048 \
  two_stage_top_p=0.85 \
  two_stage_hc_temperature=1.0 \
  two_stage_lc_temperature=0.6 \
  two_stage_hc_max_candidates=3 \
  two_stage_lc_max_candidates=5 \
  two_stage_fallback='majority' \
  two_stage_fallback_mode='no_update_both' \
  +algorithm.lam_div=0.05 \
  +algorithm.c_max=2 \
  +algorithm.div_sc_threshold=0.6 \
  trainer.logger=['console','wandb'] \
  trainer.project_name=$WANDB_PROJECT \
  trainer.experiment_name=$LOG_NAME \
  trainer.n_gpus_per_node=4 \
  trainer.nnodes=1 \
  trainer.save_freq=15 \
  trainer.test_freq=5 \
  trainer.max_actor_ckpt_to_keep=0 \
  trainer.max_critic_ckpt_to_keep=0 \
  trainer.default_local_dir=$OUTPUT_DIR \
  trainer.total_epochs=$EPISODE "$@"

echo "=== Training Completed ==="
echo "Output directory: $OUTPUT_DIR"
echo "Project name: $WANDB_PROJECT"
echo "Experiment name: $LOG_NAME"