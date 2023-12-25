export CUDA_VISIBLE_DEVICES=0,1,2,3
export WANDB_PROJECT=SynTableQA
export WANDB_ENTITY=siyue-zhang

model_name="microsoft/tapex-large-finetuned-tabfact"
run_name="squall_selector_l_token_tabfact"
dataset_name="squall"
output_dir="output/squall_selector_l_token_tabfact"

python ./train.py \
  --do_train \
  --do_eval \
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
  --per_device_train_batch_size 3 \
  --per_device_eval_batch_size 4 \
  --gradient_accumulation_steps 4 \
  --learning_rate 2e-5 \
  --weight_decay 0.01 \
  --max_grad_norm 0.1 \
  --predict_with_generate \
  --save_steps 50 \
  --save_total_limit 2 \
  --logging_steps 10 \
  --warmup_ratio 0.1 \
  --evaluation_strategy steps \
  --eval_steps 50 \
  --add_continue_token 


  # --max_eval_samples 200 \
  # --max_train_samples 100
