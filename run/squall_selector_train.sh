export CUDA_VISIBLE_DEVICES=0,1
export WANDB_PROJECT=Selector
export WANDB_ENTITY=siyue-zhang

model_name="neulab/omnitab-large"
run_name="squall_plus_selector_omnitab_aug"
dataset_name="squall"
output_dir="output/squall_plus_selector_omnitab_aug"

python ./run.py \
  --do_train \
  --do_eval \
  --fp16 \
  --num_train_epochs 50 \
  --run_name ${run_name} \
  --task selector \
  --test_split 1 \
  --output_dir ${output_dir} \
  --model_name_or_path ${model_name} \
  --overwrite_output_dir \
  --load_best_model_at_end \
  --metric_for_best_model acc \
  --max_source_length 1024 \
  --max_target_length 128 \
  --dataset_name ${dataset_name} \
  --per_device_train_batch_size 4 \
  --per_device_eval_batch_size 6 \
  --gradient_accumulation_steps 8 \
  --learning_rate 2e-5 \
  --predict_with_generate \
  --save_steps 50 \
  --save_total_limit 2 \
  --logging_steps 10 \
  --warmup_ratio 0.1 \
  --evaluation_strategy steps \
  --eval_steps 50

  # --resume_from_checkpoint output/squall_plus_selector_omnitab_aug/checkpoint-650 \
  # --max_eval_samples 200 \
  # --max_train_samples 100
  # --resume_from_checkpoint output/squall_tableqa1/checkpoint-${checkpoint} \


