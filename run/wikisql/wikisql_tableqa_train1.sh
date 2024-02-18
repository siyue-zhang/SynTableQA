export CUDA_VISIBLE_DEVICES=0,1
export WANDB_PROJECT=STQA_wikisql
export WANDB_ENTITY=siyue-zhang

model_name="microsoft/tapex-large"
run_name="wikisql_tableqa1"
dataset_name="wikisql"
output_dir="output/wikisql_tableqa1"

python ./run.py \
  --do_train \
  --do_eval \
  --dataset_name ${dataset_name} \
  --split_id 1 \
  --model_name_or_path ${model_name} \
  --max_source_length 1024 \
  --max_target_length 128 \
  --val_max_target_length 128 \
  --overwrite_output_dir \
  --load_best_model_at_end \
  --metric_for_best_model acc \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 4 \
  --per_device_eval_batch_size 4 \
  --num_train_epochs 100 \
  --warmup_ratio 0.1 \
  --learning_rate 3e-5 \
  --weight_decay 0.01 \
  --label_smoothing_factor 0.1 \
  --logging_steps 10 \
  --evaluation_strategy steps \
  --predict_with_generate \
  --eval_steps 50 \
  --save_steps 50 \
  --num_beams 5 \
  --generation_max_length 128 \
  --run_name ${run_name} \
  --task tableqa \
  --output_dir ${output_dir} \
  --save_total_limit 1