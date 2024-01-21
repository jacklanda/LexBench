inp=$1
openie=$2

python identify_ncs.py ${inp} ${inp}.pnc

python run.py --model_name_or_path t5-base --output_dir models/uniGen_randSplit_sentence/ --overwrite_output_dir --do_predict --predict_with_generate --overwrite_cache --metric_type sacrebleu --num_train_epochs 10  --per_device_train_batch_size 32 --data_type sentence --seed 108 --model_type uniGen --test_file ${inp}.pnc --output_prediction_file ${inp}.pnci

python integrate_input.py ${inp}.pnci ${inp}.exts ${inp}.int_inp
python run.py --model_name_or_path t5-base --output_dir models/integrate_pnc/ --overwrite_output_dir --do_predict --predict_with_generate --overwrite_cache --metric_type accuracy --num_train_epochs 150  --per_device_train_batch_size 16 --data_type integrate --model_type uniGen --test_file ${inp}.int_inp  --output_prediction_file ${inp}.int_out

python integrate_output.py ${inp}.int_out ${inp}.new_exts
python consolidate_exts.py ${inp}.new_exts ${inp}.exts ${inp}.nci.exts