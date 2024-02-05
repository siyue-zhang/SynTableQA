export CUDA_VISIBLE_DEVICES=0
export WANDB_PROJECT=SynTableQA
export WANDB_ENTITY=siyue-zhang

model_name="neulab/omnitab-large"
dataset_name="spider"
output_dir="output/spider_syn_selector"
checkpoint=2100


python ./run.py \
  --task selector \
  --test_split 1 \
  --do_predict \
  --spider_syn True \
  --predict_split test \
  --output_dir ${output_dir} \
  --resume_from_checkpoint ${output_dir}/checkpoint-${checkpoint} \
  --model_name_or_path ${model_name} \
  --max_source_length 1024 \
  --per_device_eval_batch_size 8 \
  --dataset_name ${dataset_name} \
  --predict_with_generate 

  # --squall_downsize 10
#   --max_predict_samples 20
