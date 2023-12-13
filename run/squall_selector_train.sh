export CUDA_VISIBLE_DEVICES=2
export WANDB_PROJECT=SynTableQA
export WANDB_ENTITY=siyue-zhang

model_name="microsoft/tapex-base-finetuned-wtq"
run_name="squall_selector_tapex_large"
dataset_name="squall"
output_dir="output/squall_selector"

python ./train.py \
  --do_train \
  --do_eval \
  --num_train_epochs 30 \
  --run_name ${run_name} \
  --task selector \
  --output_dir ${output_dir} \
  --model_name_or_path ${model_name} \
  --overwrite_output_dir \
  --max_source_length 1024 \
  --max_target_length 128 \
  --dataset_name ${dataset_name} \
  --per_device_train_batch_size 4 \
  --per_device_eval_batch_size 4 \
  --gradient_accumulation_steps 6 \
  --learning_rate 3e-5 \
  --weight_decay 0.01 \
  --max_grad_norm 0.1 \
  --predict_with_generate \
  --save_steps 200 \
  --save_total_limit 1 \
  --logging_steps 10 \
  --warmup_steps 200 \
  --evaluation_strategy steps \
  --eval_steps 50 \
  --max_eval_samples 10 \
  --max_train_samples 100

# --resume_from_checkpoint ${output_dir}/checkpoint-${checkpoint} \
