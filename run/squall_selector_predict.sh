export CUDA_VISIBLE_DEVICES=0
export WANDB_PROJECT=SynTableQA
export WANDB_ENTITY=siyue-zhang

model_name="neulab/omnitab-large"
dataset_name="squall"
# output_dir="output/squall_plus_selector_omnitab"
# checkpoint=2150
output_dir="squall_selector_tableqa_omnitab"
checkpoint=150

python ./train.py \
  --task selector \
  --model tableqa \
  --test_split 1 \
  --do_predict \
  --predict_split dev \
  --output_dir ${output_dir} \
  --resume_from_checkpoint ${output_dir}/checkpoint-${checkpoint} \
  --model_name_or_path ${model_name} \
  --max_source_length 1024 \
  --per_device_eval_batch_size 8 \
  --dataset_name ${dataset_name} \
  --predict_with_generate \
  --generation_max_length 128 


#   --max_predict_samples 20
