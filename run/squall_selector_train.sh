export CUDA_VISIBLE_DEVICES=0
export WANDB_PROJECT=SynTableQA
export WANDB_ENTITY=siyue-zhang

model_name="microsoft/tapex-base-finetuned-tabfact"
run_name="squall_selector"
dataset_name="squall"
output_dir="output/squall_selector"

python ./train.py \
  --do_train \
  --do_eval \
  --num_train_epochs 10 \
  --run_name ${run_name} \
  --task selector \
  --output_dir ${output_dir} \
  --model_name_or_path ${model_name} \
  --overwrite_output_dir \
  --load_best_model_at_end \
  --metric_for_best_model acc \
  --max_source_length 1024 \
  --max_target_length 128 \
  --dataset_name ${dataset_name} \
  --per_device_train_batch_size 4 \
  --per_device_eval_batch_size 8 \
  --gradient_accumulation_steps 4 \
  --learning_rate 3e-5 \
  --weight_decay 0.01 \
  --max_grad_norm 0.1 \
  --predict_with_generate \
  --save_steps 50 \
  --save_total_limit 2 \
  --logging_steps 10 \
  --warmup_ratio 0.2 \
  --evaluation_strategy steps \
  --eval_steps 50 

  # --add_from_train \
  # --max_eval_samples 200 \
  # --max_train_samples 100
