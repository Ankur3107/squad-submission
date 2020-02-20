CUDA_VISIBLE_DEVICES=2 python predict_squad.py \
  --predict_file=dev-vs2.0.json \
  --model_dir=../tuned_albert/ \
  --model_name="tuned_albert" \
  --null_score_diff_threshold=1.0 \
  --checkpoint_path=ctl_step_6992.ckpt-3

CUDA_VISIBLE_DEVICES=2 python predict_squad.py \
  --predict_file=dev-vs2.0.json \
  --predict_batch_size=24 \
  --model_dir=../albert_bidaf/ \
  --model_name=tuned_albert \
  --checkpoint_path=ctl_step_5508.ckpt-1,ctl_step_6992.ckpt-3


CUDA_VISIBLE_DEVICES=2 python predict_squad.py \
  --predict_file=/tf/Ankur_Workspace/albert/ALBERT-TF2.0/SQuAD/dev-v2.0.json \
  --predict_batch_size=24 \
  --model_dir=../albert_bidaf/,../tuned_albert/ \
  --model_name=tuned_albert \
  --checkpoint_path=ctl_step_5508.ckpt-1,ctl_step_6992.ckpt-3

python ensemble.py \
  --input_nbest_files=../tuned_albert/nbest_predictions.json,../albert_bidaf/nbest_predictions.json \
  --input_null_files=../tuned_albert/null_odds.json,../albert_bidaf/null_odds.json \
  --null_score_thresh=0.0 \
  --output_file=../predictions.json