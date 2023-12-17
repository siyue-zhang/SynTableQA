export CUDA_VISIBLE_DEVICES=0,1,2,3
export WANDB_PROJECT=SynTableQA
export WANDB_ENTITY=siyue-zhang

model_name="neulab/omnitab-large"
run_name="squall_plus_tableqa"
dataset_name="squall"
output_dir="output/squall_plus_tableqa"

python ./train.py \
  --do_train \
  --do_eval \
  --num_train_epochs 50 \
  --run_name ${run_name} \
  --task tableqa \
  --squall_plus plus \
  --output_dir ${output_dir} \
  --model_name_or_path ${model_name} \
  --overwrite_output_dir \
  --load_best_model_at_end \
  --metric_for_best_model acc \
  --max_source_length 1024 \
  --max_target_length 128 \
  --dataset_name ${dataset_name} \
  --per_device_train_batch_size 6 \
  --per_device_eval_batch_size 4 \
  --gradient_accumulation_steps 6 \
  --learning_rate 2e-5 \
  --fp16 \
  --predict_with_generate \
  --generation_max_length 128 \
  --num_beams 5 \
  --save_total_limit 2 \
  --logging_steps 10 \
  --warmup_ratio 0.1 \
  --evaluation_strategy steps \
  --save_steps 200 \
  --eval_steps 100

  # --max_eval_samples 10 \
  # --max_train_samples 100

