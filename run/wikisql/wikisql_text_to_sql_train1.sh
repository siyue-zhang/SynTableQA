export CUDA_VISIBLE_DEVICES=0,1
export WANDB_PROJECT=STQA_wikisql
export WANDB_ENTITY=siyue-zhang

model_name="t5-large"
run_name="wikisql_text_to_sql1_R"
dataset_name="wikisql"
output_dir="output/wikisql_text_to_sql1_R"

python ./run.py \
  --do_train \
  --do_eval \
  --num_train_epochs 10 \
  --run_name ${run_name} \
  --task text_to_sql \
  --output_dir ${output_dir} \
  --model_name_or_path ${model_name} \
  --overwrite_output_dir \
  --load_best_model_at_end \
  --metric_for_best_model acc \
  --max_source_length 1024 \
  --max_target_length 128 \
  --dataset_name ${dataset_name} \
  --split_id 1 \
  --per_device_train_batch_size 4 \
  --per_device_eval_batch_size 8 \
  --gradient_accumulation_steps 16 \
  --postproc_fuzzy_string True \
  --learning_rate 3e-4 \
  --weight_decay 0.01 \
  --predict_with_generate \
  --generation_max_length 128 \
  --num_beams 5 \
  --save_total_limit 1 \
  --logging_steps 10 \
  --warmup_ratio 0.1 \
  --evaluation_strategy steps \
  --save_steps 100 \
  --eval_steps 100 \
  --max_eval_samples 2000
  # --max_train_samples 100 \


