export PYTHONPATH=${PYTHONPATH}:`pwd`; mkdir full_code; cp model/ctl_step_6992.ckpt-3.data-00001-of-00002 full_code/; cp -r code/ full_code/; cd full_code; python predict_squad.py --predict_file ../dev-v2.0.json

mkdir full_code; cp model/ctl_step_6992.ckpt-3.data-00001-of-00002 full_code/; cp -r code/ full_code/; cd full_code

cl run 	dev-v2.0.json:dev-v2.0.json :code :model "mkdir full_code; cp -r code/ full_code/; cp model/ctl_step_6992.ckpt-3.data-00001-of-00002 full_code/code/; cp dev-v2.0.json full_code/code/; cd full_code/code; python predict_squad.py --predict_file dev-v2.0.json" -n run-predictions-3 --request-docker-image ankur310794/tensorflow:tfv2.0 --request-memory 8g --request-cpus 2 --request-gpus 1

cl run 	dev-v2.0.json:dev-v2.0.json :code :model "mkdir full_code; cp -r code/ full_code/; cp model/ctl_step_6992.ckpt-3.data-00001-of-00002 full_code/code/; cp dev-v2.0.json full_code/code/; cd full_code/code; python predict_squad.py --predict_file dev-v2.0.json --model_dir ../../" -n run-predictions-dev --request-docker-image ankur310794/tensorflow:tfv2.0 --request-memory 8g --request-cpus 2 --request-gpus 1

cl run 	dev-v2.0.json:dev-v2.0.json :code :model "mkdir full_code; cp -r code/ full_code/; cp model/ctl_step_6992.ckpt-3.data-00001-of-00002 full_code/code/; cp dev-v2.0.json full_code/code/; cd full_code/code; python predict_squad.py --predict_file dev-v2.0.json --model_dir ../../ --strategy_type mirror --predict_batch_size 16" -n run-predictions-dev --request-docker-image ankur310794/tensorflow:tfv2.0 --request-memory 10g --request-cpus 2 --request-gpus 2

cl run 	dev-vs2.0.json:dev-vs2.0.json :code :model "mkdir full_code; cp -r code/ full_code/; cp model/ctl_step_6992.ckpt-3.data-00001-of-00002 full_code/code/; cp dev-vs2.0.json full_code/code/; cd full_code/code; python predict_squad.py --predict_file dev-vs2.0.json" -n run-predictions-3 --request-docker-image ankur310794/tensorflow:latest --request-memory 8g --request-cpus 2 --request-gpus 1

cl run 	dev-vs2.0.json:dev-vs2.0.json :code :model "mkdir full_code; cp -r code/ full_code/; cp model/ctl_step_6992.ckpt-3.data-00001-of-00002 full_code/code/; cp dev-vs2.0.json full_code/code/; cd full_code/code; python predict_squad.py --predict_file dev-vs2.0.json --model_dir ../../; cd ../../; rm -rf full_code" -n run-predictions-3 --request-docker-image ankur310794/tensorflow:latest --request-memory 8g --request-cpus 2 --request-gpus 1



cl run dev-v2.0.json:dev-v2.0.json code:code :model :model2 ensemble.sh:ensemble.sh "bash ensemble.sh dev-v2.0.json predictions.json" -n run-predictions-ensemble --request-docker-image ankur310794/tensorflow:tfv2.0 --request-memory 8g --request-cpus 2 --request-gpus 1

cl run dev-vs2.0.json:dev-vs2.0.json code:code :model :model2 ensemble.sh:ensemble.sh "bash ensemble.sh dev-vs2.0.json predictions.json" -n run-predictions-ensemble --request-docker-image ankur310794/tensorflow:latest --request-memory 8g --request-cpus 1

cl run google_drive_2.sh:google_drive_2.sh "sh google_drive_2.sh" -n model2 --request-docker-image codalab/default-cpu:latest --request-memory 4g --request-cpus 1 --request-network true