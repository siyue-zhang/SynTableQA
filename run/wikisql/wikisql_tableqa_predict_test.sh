export CUDA_VISIBLE_DEVICES=0,1,2,3

model_name="yilunzhao/omnitab-large-finetuned-wikisql"
dataset_name="wikisql"
output_dir="output/omnitab-large-finetuned-wikisql"


python ./run.py \
  --task tableqa \
  --do_predict \
  --predict_split test \
  --perturbation_type original \
  --output_dir ${output_dir} \
  --model_name_or_path ${model_name} \
  --max_source_length 1024 \
  --max_target_length 128 \
  --per_device_eval_batch_size 2 \
  --dataset_name ${dataset_name} \
  --split_id 1 \
  --predict_with_generate \
  --generation_max_length 128 \
  --num_beams 5 
  # --max_predict_samples 2000


