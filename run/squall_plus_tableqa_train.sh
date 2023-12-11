export CUDA_VISIBLE_DEVICES=1,3
export WANDB_PROJECT=SynTableQA
export WANDB_ENTITY=siyue-zhang

model_name="neulab/omnitab-large"
run_name="squall_plus_tableqa"
dataset_name="squall"
output_dir="output/squall_plus_tableqa"

python ./train_expert.py \
  --do_train \
  --do_eval \
  --num_train_epochs 30 \
  --run_name ${run_name} \
  --output_dir ${output_dir} \
  --model_name_or_path ${model_name} \
  --overwrite_output_dir \
  --max_source_length 1024 \
  --max_target_length 256 \
  --dataset_name ${dataset_name} \
  --squall_plus plus \
  --per_device_train_batch_size 8 \
  --per_device_eval_batch_size 8 \
  --gradient_accumulation_steps 6 \
  --learning_rate 2e-5 \
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
  --max_train_samples 100 \

# --resume_from_checkpoint ${output_dir}/checkpoint-${checkpoint} \
