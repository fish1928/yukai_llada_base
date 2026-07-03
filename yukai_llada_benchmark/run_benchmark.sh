#!/bin/bash

accelerate launch --num_processes=1 run_benchmark_main.py --tasks ifeval --limit 10 --model test --batch_size 1 --num_fewshot 1 --model_args id_model='GSAI-ML/LLaDA-8B-Base',size_batch=1,len_target=64,num_blocks=1,num_unmask_per_step=1,id_mask=126336,device='cuda:0',step_refresh_remainder=4,select_only_in_h=True,runner=run_model_semi_cached_mlp,h=5 || echo "command failed, continuing"


accelerate launch --num_processes=1 run_benchmark_main.py --tasks ifeval --limit 10 --model test --batch_size 1 --num_fewshot 1 --model_args id_model='GSAI-ML/LLaDA-8B-Base',size_batch=1,len_target=64,num_blocks=1,num_unmask_per_step=1,id_mask=126336,device='cuda:0',step_refresh_remainder=16,select_only_in_h=True,runner=run_model_semi_cached_mlp,h=16 || echo "command failed, continuing"


accelerate launch --num_processes=1 run_benchmark_main.py --tasks ifeval --limit 10 --model test --batch_size 1 --num_fewshot 1 --model_args id_model='GSAI-ML/LLaDA-8B-Base',size_batch=1,len_target=64,num_blocks=1,num_unmask_per_step=1,id_mask=126336,device='cuda:0',step_refresh_remainder=4,select_only_in_h=True,runner=run_model_semi_cached_mlp,h=16 || echo "command failed, continuing"


accelerate launch --num_processes=1 run_benchmark_main.py --tasks ifeval --limit 10 --model test --batch_size 1 --num_fewshot 1 --model_args id_model='GSAI-ML/LLaDA-8B-Base',size_batch=1,len_target=64,num_blocks=1,num_unmask_per_step=1,id_mask=126336,device='cuda:0',step_refresh_remainder=64,select_only_in_h=True,runner=run_model_semi_cached_mlp,h=16 || echo "command failed, continuing"


#accelerate launch --num_processes=1 run_benchmark_main.py --tasks gsm8k --limit 50 --model test --batch_size 1 --num_fewshot 1 --model_args id_model='GSAI-ML/LLaDA-8B-Base',size_batch=1,len_target=128,num_blocks=2,num_unmask_per_step=1,id_mask=126336,device='cuda:0',step_refresh_remainder=4,select_only_in_h=True,runner=run_model_semi_cached_mlp || echo "command failed, continuing"


#accelerate launch --num_processes=1 run_benchmark_main.py --tasks gsm8k --limit 50 --model test --batch_size 1 --num_fewshot 1 --model_args id_model='GSAI-ML/LLaDA-8B-Base',size_batch=1,len_target=128,num_blocks=2,num_unmask_per_step=1,id_mask=126336,device='cuda:0',step_refresh_remainder=2,select_only_in_h=True,runner=run_model_semi_cached_mlp || echo "command failed, continuing"


# run llada
accelerate launch --num_processes=1 run_benchmark_main.py --tasks ifeval --limit 10 --model test --batch_size 1 --num_fewshot 1 --model_args id_model='GSAI-ML/LLaDA-8B-Base',size_batch=1,len_target=64,num_blocks=1,num_unmask_per_step=1,id_mask=126336,device='cuda:0',runner=run_model_semi || echo "command failed, continuing"


# run llada semi cached
accelerate launch --num_processes=1 run_benchmark_main.py --tasks ifeval --limit 10 --model test --batch_size 1 --num_fewshot 1 --model_args id_model='GSAI-ML/LLaDA-8B-Base',size_batch=1,len_target=64,num_blocks=1,num_unmask_per_step=1,id_mask=126336,device='cuda:0',runner=run_model_semi_cached || echo "command failed, continuing"


# run dllm
accelerate launch --num_processes=1 run_benchmark_main.py --tasks ifeval --limit 10 --model test --batch_size 1 --num_fewshot 1 --model_args id_model='GSAI-ML/LLaDA-8B-Base',size_batch=1,len_target=64,num_blocks=1,num_unmask_per_step=1,id_mask=126336,device='cuda:0',runner=run_model_dllm
