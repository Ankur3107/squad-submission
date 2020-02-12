export PYTHONPATH=${PYTHONPATH}:`pwd`; mkdir full_code; cp model/ctl_step_6992.ckpt-3.data-00001-of-00002 full_code/; cp -r code/ full_code/; cd full_code; python predict_squad.py --predict_file ../dev-v2.0.json

mkdir full_code; cp model/ctl_step_6992.ckpt-3.data-00001-of-00002 full_code/; cp -r code/ full_code/; cd full_code

cl run 	dev-v2.0.json:dev-v2.0.json :code :model "export PYTHONPATH=${PYTHONPATH}:`pwd`; mkdir full_code; cp -r code/ full_code/; cp model/ctl_step_6992.ckpt-3.data-00001-of-00002 full_code/code/; cp dev-v2.0.json full_code/code/; cd full_code/code; python predict_squad.py --predict_file dev-v2.0.json" -n run-predictions-3 --request-docker-image mingdachengoogle/tf_sp_gpu:latest --request-memory 8g --request-cpus 2 --request-gpus 1