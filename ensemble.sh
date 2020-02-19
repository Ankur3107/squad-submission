#!/bin/bash

set -e
set -x

export INPUT_FILE=$1
export OUTPUT_FILE=$2

mkdir full_code

mkdir tuned_albert
mkdir albert_bidaf

cp -r code/ full_code/

cp model/ctl_step_6992.ckpt-3.data-00001-of-00002 full_code/code/
cp model2/ctl_step_8262.ckpt-1.data-00001-of-00002 full_code/code/


cp dev-vs2.0.json full_code/code/

cd full_code/code

python predict_squad.py \
  --predict_file=${INPUT_FILE} \
  --model_dir=../../tuned_albert/ \
  --model_name=tuned_albert \
  --null_score_diff_threshold=1.0 \
  --checkpoint_path=ctl_step_6992.ckpt-3

python predict_squad.py \
  --predict_file=${INPUT_FILE} \
  --model_dir=../../albert_bidaf/ \
  --model_name=albert_bidaf \
  --null_score_diff_threshold=-1.0 \
  --checkpoint_path=ctl_step_8262.ckpt-1

python ensemble.py \
  --input_nbest_files=../../tuned_albert/nbest_predictions.json,../../albert_bidaf/nbest_predictions.json \
  --input_null_files=../../tuned_albert/null_odds.json,../../albert_bidaf/null_odds.json \
  --null_score_thresh=0.0 \
  --output_file=${OUTPUT_FILE}


cd ../../

rm -rf full_code