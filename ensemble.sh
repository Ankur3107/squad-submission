#!/bin/bash

set -e
set -x

export INPUT_FILE=$1
export OUTPUT_FILE=$2

mkdir full_code

mkdir tuned_albert
mkdir tuned_albert_2

cp -r code/ full_code/

cp model/ctl_step_6992.ckpt-3.data-00001-of-00002 full_code/code/
cp model2/ctl_step_5508.ckpt-1.data-00001-of-00002 full_code/code/


cp ${INPUT_FILE} full_code/code/

cd full_code/code

python predict_squad.py \
  --predict_file=${INPUT_FILE} \
  --model_dir=../../tuned_albert/,../../tuned_albert_2/ \
  --model_name=tuned_albert \
  --null_score_diff_threshold=1.0 \
  --checkpoint_path=ctl_step_6992.ckpt-3,ctl_step_5508.ckpt-1

python ensemble.py \
  --input_nbest_files=../../tuned_albert/nbest_predictions.json,../../tuned_albert_2/nbest_predictions.json \
  --input_null_files=../../tuned_albert/null_odds.json,../../tuned_albert_2/null_odds.json \
  --null_score_thresh=0.0 \
  --output_file=../../${OUTPUT_FILE}


cd ../../

rm -rf full_code