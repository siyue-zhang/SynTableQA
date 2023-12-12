export CUDA_VISIBLE_DEVICES=1,3
export WANDB_PROJECT=SynTableQA
export WANDB_ENTITY=siyue-zhang

model_name="t5-large"
run_name="squall_text_to_sql"
dataset_name="squall"
output_dir="output/squall_text_to_sql"

python ./train.py \
  --do_train \
  --do_eval \
  --num_train_epochs 30 \
  --run_name ${run_name} \
  --task text_to_sql \
  --output_dir ${output_dir} \
  --model_name_or_path ${model_name} \
  --overwrite_output_dir \
  --max_source_length 1024 \
  --max_target_length 256 \
  --dataset_name ${dataset_name} \
  --per_device_train_batch_size 4 \
  --per_device_eval_batch_size 8 \
  --gradient_accumulation_steps 16 \
  --postproc_fuzzy_string True \
  --learning_rate 3e-4 \
  --weight_decay 0.01 \
  --predict_with_generate \
  --generation_max_length 128 \
  --num_beams 5 \
  --save_steps 500 \
  --save_total_limit 1 \
  --logging_steps 10 \
  --warmup_steps 200 \
  --evaluation_strategy steps \
  --eval_steps 50 \
  --max_eval_samples 10 \
  --max_train_samples 1000 \

# --resume_from_checkpoint ${output_dir}/checkpoint-${checkpoint} \
