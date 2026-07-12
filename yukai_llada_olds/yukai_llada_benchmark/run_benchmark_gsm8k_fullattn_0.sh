#!/bin/bash

# mlp-1B k=4,h=5
accelerate launch --num_processes=1 run_benchmark_main.py --tasks gsm8k --limit 50 --model test --batch_size 1 --num_fewshot 5 --model_args id_model='GSAI-ML/LLaDA-8B-Base',size_batch=1,len_target=128,num_blocks=1,num_unmask_per_step=1,id_mask=126336,device='cuda:1',step_refresh_remainder=4,select_only_in_h=True,runner=run_model_semi_cached_mlp_fullattn,h=5 || echo "command failed, continuing"


# mlp-1B k=8,h=5
accelerate launch --num_processes=1 run_benchmark_main.py --tasks gsm8k --limit 50 --model test --batch_size 1 --num_fewshot 5 --model_args id_model='GSAI-ML/LLaDA-8B-Base',size_batch=1,len_target=128,num_blocks=1,num_unmask_per_step=1,id_mask=126336,device='cuda:1',step_refresh_remainder=8,select_only_in_h=True,runner=run_model_semi_cached_mlp_fullattn,h=5 || echo "command failed, continuing"


# mlp-1B k=12,h=5
accelerate launch --num_processes=1 run_benchmark_main.py --tasks gsm8k --limit 50 --model test --batch_size 1 --num_fewshot 5 --model_args id_model='GSAI-ML/LLaDA-8B-Base',size_batch=1,len_target=128,num_blocks=1,num_unmask_per_step=1,id_mask=126336,device='cuda:1',step_refresh_remainder=12,select_only_in_h=True,runner=run_model_semi_cached_mlp_fullattn,h=5 || echo "command failed, continuing"


# mlp-1B k=16,h=5
accelerate launch --num_processes=1 run_benchmark_main.py --tasks gsm8k --limit 50 --model test --batch_size 1 --num_fewshot 5 --model_args id_model='GSAI-ML/LLaDA-8B-Base',size_batch=1,len_target=128,num_blocks=1,num_unmask_per_step=1,id_mask=126336,device='cuda:1',step_refresh_remainder=16,select_only_in_h=True,runner=run_model_semi_cached_mlp_fullattn,h=5 || echo "command failed, continuing"


# mlp-1B k=4,h=8
accelerate launch --num_processes=1 run_benchmark_main.py --tasks gsm8k --limit 50 --model test --batch_size 1 --num_fewshot 5 --model_args id_model='GSAI-ML/LLaDA-8B-Base',size_batch=1,len_target=128,num_blocks=1,num_unmask_per_step=1,id_mask=126336,device='cuda:1',step_refresh_remainder=4,select_only_in_h=True,runner=run_model_semi_cached_mlp_fullattn,h=8 || echo "command failed, continuing"


# mlp-1B k=4,h=12
accelerate launch --num_processes=1 run_benchmark_main.py --tasks gsm8k --limit 50 --model test --batch_size 1 --num_fewshot 5 --model_args id_model='GSAI-ML/LLaDA-8B-Base',size_batch=1,len_target=128,num_blocks=1,num_unmask_per_step=1,id_mask=126336,device='cuda:1',step_refresh_remainder=4,select_only_in_h=True,runner=run_model_semi_cached_mlp_fullattn,h=12 || echo "command failed, continuing"


# mlp-1B k=4,h=16
accelerate launch --num_processes=1 run_benchmark_main.py --tasks gsm8k --limit 50 --model test --batch_size 1 --num_fewshot 5 --model_args id_model='GSAI-ML/LLaDA-8B-Base',size_batch=1,len_target=128,num_blocks=1,num_unmask_per_step=1,id_mask=126336,device='cuda:1',step_refresh_remainder=4,select_only_in_h=True,runner=run_model_semi_cached_mlp_fullattn,h=16 || echo "command failed, continuing"
