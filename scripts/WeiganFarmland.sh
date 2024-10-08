python -u run.py  --task_name classification  --is_training 1  --root_path ./data/WeiganFarmland/  --model_id WeiganFarmland  --model Reformer  --data UEA  --e_layers 3  --batch_size 16  --d_model 128  --d_ff 256  --top_k 3  --des 'Exp'  --itr 1  --learning_rate 0.001  --train_epochs 100  --patience 10

python -u run.py  --task_name classification  --is_training 0  --root_path ./data/WeiganFarmland/  --model_id WeiganFarmland  --model Reformer  --data UEA  --e_layers 3  --batch_size 16  --d_model 128  --d_ff 256  --top_k 3  --des 'Exp'  --itr 1  --learning_rate 0.001  --train_epochs 100  --patience 10

python -u run.py --task_name classification --is_training 1 --root_path ./data/WeiganFarmland --model_id WeiganFarmland --model TimesNet --data UEA --e_layers 2 --batch_size 16 --d_model 32 --d_ff 64 --top_k 3 --des 'Exp' --itr 1 --learning_rate 0.001 --train_epochs 30 --patience 10

python -u run.py --task_name classification --is_training 0 --root_path ./data/WeiganFarmland --model_id WeiganFarmland --model TimesNet --data UEA --e_layers 2 --batch_size 16 --d_model 32 --d_ff 64 --top_k 3 --des 'Exp' --itr 1 --learning_rate 0.001 --train_epochs 30 --patience 10

python -u run.py --task_name classification --is_training 1 --root_path ./data/WeiganFarmland --model_id WeiganFarmland --model DLinear --data UEA --e_layers 3 --batch_size 16 --d_model 128 --d_ff 256 --top_k 3 --des 'Exp' --itr 1 --learning_rate 0.001 --train_epochs 30 --patience 10

python -u run.py --task_name classification --is_training 0 --root_path ./data/WeiganFarmland --model_id WeiganFarmland --model DLinear --data UEA --e_layers 3 --batch_size 16 --d_model 128 --d_ff 256 --top_k 3 --des 'Exp' --itr 1 --learning_rate 0.001 --train_epochs 30 --patience 10

python -u run.py --task_name classification --is_training 1 --root_path ./data/WeiganFarmland --model_id WeiganFarmland --model LightTS --data UEA --e_layers 3 --batch_size 16 --d_model 128 --d_ff 256 --top_k 3 --des 'Exp' --itr 1 --learning_rate 0.001 --train_epochs 30 --patience 10

python -u run.py --task_name classification --is_training 0 --root_path ./data/WeiganFarmland --model_id WeiganFarmland --model LightTS --data UEA --e_layers 3 --batch_size 16 --d_model 128 --d_ff 256 --top_k 3 --des 'Exp' --itr 1 --learning_rate 0.001 --train_epochs 30 --patience 10

python -u run.py --task_name classification --is_training 1 --root_path ./data/WeiganFarmland --model_id WeiganFarmland --model Transformer --data UEA --e_layers 3 --batch_size 16 --d_model 128 --d_ff 256 --top_k 3 --des 'Exp' --itr 1 --learning_rate 0.001 --train_epochs 30 --patience 10

python -u run.py --task_name classification --is_training 0 --root_path ./data/WeiganFarmland --model_id WeiganFarmland --model Transformer --data UEA --e_layers 3 --batch_size 16 --d_model 128 --d_ff 256 --top_k 3 --des 'Exp' --itr 1 --learning_rate 0.001 --train_epochs 30 --patience 10

python -u run.py --task_name classification --is_training 1 --root_path ./data/WeiganFarmland --model_id WeiganFarmland --model FiLM --data UEA --e_layers 3 --batch_size 16 --d_model 128 --d_ff 256 --top_k 3 --des 'Exp' --itr 1 --learning_rate 0.001 --train_epochs 30 --patience 10

python -u run.py --task_name classification --is_training 0 --root_path ./data/WeiganFarmland --model_id WeiganFarmland --model FiLM --data UEA --e_layers 3 --batch_size 16 --d_model 128 --d_ff 256 --top_k 3 --des 'Exp' --itr 1 --learning_rate 0.001 --train_epochs 30 --patience 10

python -u run.py --task_name classification --is_training 1 --root_path ./data/WeiganFarmland --model_id WeiganFarmland --model PatchTST --data UEA --e_layers 3 --batch_size 16 --d_model 128 --d_ff 256 --top_k 3 --des 'Exp' --itr 1 --learning_rate 0.001 --train_epochs 30 --patience 10

python -u run.py --task_name classification --is_training 0 --root_path ./data/WeiganFarmland --model_id WeiganFarmland --model PatchTST --data UEA --e_layers 3 --batch_size 16 --d_model 128 --d_ff 256 --top_k 3 --des 'Exp' --itr 1 --learning_rate 0.001 --train_epochs 30 --patience 10

python -u run.py --task_name classification --is_training 1 --root_path ./data/WeiganFarmland --model_id WeiganFarmland --model TimesNet --data UEA --e_layers 3 --batch_size 16 --d_model 32 --d_ff 32 --top_k 3 --des 'Exp' --itr 1 --learning_rate 0.001 --train_epochs 30 --patience 10

python -u run.py --task_name classification --is_training 0 --root_path ./data/WeiganFarmland --model_id WeiganFarmland --model TimesNet --data UEA --e_layers 3 --batch_size 16 --d_model 32 --d_ff 32 --top_k 3 --des 'Exp' --itr 1 --learning_rate 0.001 --train_epochs 30 --patience 10

python -u run.py --task_name classification --is_training 1 --root_path ./data/WeiganFarmland --model_id WeiganFarmland --model FEDformer --data UEA --e_layers 3 --batch_size 16 --d_model 32 --d_ff 32 --top_k 3 --des 'Exp' --itr 1 --learning_rate 0.001 --train_epochs 30 --patience 10

python -u run.py --task_name classification --is_training 0 --root_path ./data/WeiganFarmland --model_id WeiganFarmland --model FEDformer --data UEA --e_layers 3 --batch_size 16 --d_model 32 --d_ff 32 --top_k 3 --des 'Exp' --itr 1 --learning_rate 0.001 --train_epochs 30 --patience 10

python -u run.py --task_name classification --is_training 1 --root_path ./data/WeiganFarmland --model_id WeiganFarmland --model Crossformer --data UEA --e_layers 3 --batch_size 16 --d_model 32 --d_ff 32 --top_k 3 --des 'Exp' --itr 1 --learning_rate 0.001 --train_epochs 30 --patience 10

python -u run.py --task_name classification --is_training 0 --root_path ./data/WeiganFarmland --model_id WeiganFarmland --model Crossformer --data UEA --e_layers 3 --batch_size 16 --d_model 32 --d_ff 32 --top_k 3 --des 'Exp' --itr 1 --learning_rate 0.001 --train_epochs 30 --patience 10
