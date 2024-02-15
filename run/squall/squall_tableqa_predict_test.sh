export CUDA_VISIBLE_DEVICES=0

model_name="neulab/omnitab-large-finetuned-wtq"
dataset_name="squall"
output_dir="output/omnitab-large-finetuned-wtq"

python ./run.py \
  --task tableqa \
  --do_predict \
  --squall_plus True \
  --predict_split test \
  --output_dir ${output_dir} \
  --model_name_or_path ${model_name} \
  --max_source_length 1024 \
  --max_target_length 128 \
  --val_max_target_length 128 \
  --per_device_eval_batch_size 4 \
  --dataset_name ${dataset_name} \
  --split_id 1 \
  --predict_with_generate \
  --num_beams 5 

# --max_predict_samples 100