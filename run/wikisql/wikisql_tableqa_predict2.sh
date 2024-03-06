export CUDA_VISIBLE_DEVICES=0

model_name="neulab/omnitab-large"
dataset_name="wikisql"
output_dir="output/wikisql_tableqa2"
checkpoint=2900

python ./run.py \
  --task tableqa \
  --do_predict \
  --predict_split dev \
  --output_dir ${output_dir} \
  --model_name_or_path ${model_name} \
  --resume_from_checkpoint ${output_dir}/checkpoint-${checkpoint} \
  --max_source_length 1024 \
  --max_target_length 128 \
  --val_max_target_length 128 \
  --per_device_eval_batch_size 32 \
  --dataset_name ${dataset_name} \
  --split_id 2 \
  --predict_with_generate \
  --num_beams 5
# --max_predict_samples 100
# --perturbation_type original \




