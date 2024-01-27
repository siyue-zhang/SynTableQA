export CUDA_VISIBLE_DEVICES=0,1,2,3
export WANDB_PROJECT=spider_exp
export WANDB_ENTITY=siyue-zhang

model_name="neulab/omnitab-large"
run_name="spider_tableqa0"
dataset_name="spider"
output_dir="output/spider_tableqa0"

python ./train.py \
  --do_train \
  --do_eval \
  --num_train_epochs 50 \
  --run_name ${run_name} \
  --spider_syn True \
  --task tableqa \
  --output_dir ${output_dir} \
  --model_name_or_path ${model_name} \
  --overwrite_output_dir \
  --load_best_model_at_end \
  --metric_for_best_model acc \
  --max_source_length 1024 \
  --max_target_length 128 \
  --dataset_name ${dataset_name} \
  --split_id 0 \
  --per_device_train_batch_size 2 \
  --per_device_eval_batch_size 8 \
  --gradient_accumulation_steps 16 \
  --postproc_fuzzy_string True \
  --learning_rate 3e-4 \
  --weight_decay 0.01 \
  --predict_with_generate \
  --generation_max_length 128 \
  --num_beams 5 \
  --save_steps 200 \
  --save_total_limit 2 \
  --logging_steps 10 \
  --warmup_ratio 0.1 \
  --evaluation_strategy steps \
  --eval_steps 100
  # --max_eval_samples 100 
  # --max_train_samples 5000

