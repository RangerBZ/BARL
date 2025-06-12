set -x
export DS_SKIP_CUDA_CHECK=1
export CUDA_LAUNCH_BLOCKING=1
read -r -d '' training_commands <<EOF
openrlhf.cli.train_barl \
   --pretrain Qwen/Qwen2.5-Math-7B \
   --save_path ./checkpoints/qwen7b \
   --logging_steps 1 \
   --eval_steps -1 \
   --micro_train_batch_size 8 \
   --train_batch_size 128 \
   --micro_rollout_batch_size 16 \
   --rollout_batch_size 1024 \
   --max_epochs 1 \
   --prompt_max_len 512 \
   --generate_max_len 1024 \
   --zero_stage 2 \
   --bf16 \
   --actor_learning_rate 5e-7 \
   --prompt_data SynthLabsAI/Big-Math-RL-Verified \
   --input_key problem \
   --output_key answer \
   --adam_offload \
   --flash_attn \
   --gradient_checkpointing \
   --apply_chat_template \
   --n_samples_per_prompt 5
EOF

# if [[ ${1} != "slurm" ]]; then
#     deepspeed --include localhost:0,1,2,3,4,5,6,7 --module $training_commands
# fi

if [[ "${1}" != "slurm" ]]; then
    # number of available GPUs
    NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv | tail -n +2 | wc -l)

    if [[ "$NUM_GPUS" -eq 0 ]]; then
        echo "No GPUs available."
        exit 1
    fi

    GPU_LIST=$(seq -s, 0 $((NUM_GPUS - 1)))
    deepspeed --include localhost:${GPU_LIST} --module $training_commands
fi

# also needs hugging face login:
# possible conflicts: flash-attn and CUDA, build from source: pip install git+https://github.com/Dao-AILab/flash-attention.git 