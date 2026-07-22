#!/bin/bash

# mlp-1B k=16,h=4
accelerate launch --num_processes=1 run_benchmark_main.py --tasks ifeval --limit 50 --model test --batch_size 1 --num_fewshot 1 --model_args id_model='GSAI-ML/LLaDA-8B-Base',size_batch=1,len_target=256,num_blocks=1,num_unmask_per_step=1,id_mask=126336,device='cuda:0',step_refresh_remainder=16,select_only_in_h=True,runner=run_model_semi_cached_mlp,h=4 || echo "command failed, continuing"


# mlp-1B k=16,h=5
accelerate launch --num_processes=1 run_benchmark_main.py --tasks ifeval --limit 50 --model test --batch_size 1 --num_fewshot 1 --model_args id_model='GSAI-ML/LLaDA-8B-Base',size_batch=1,len_target=256,num_blocks=1,num_unmask_per_step=1,id_mask=126336,device='cuda:0',step_refresh_remainder=16,select_only_in_h=True,runner=run_model_semi_cached_mlp,h=5 || echo "command failed, continuing"


# mlp-1B k=16,h=6
accelerate launch --num_processes=1 run_benchmark_main.py --tasks ifeval --limit 50 --model test --batch_size 1 --num_fewshot 1 --model_args id_model='GSAI-ML/LLaDA-8B-Base',size_batch=1,len_target=256,num_blocks=1,num_unmask_per_step=1,id_mask=126336,device='cuda:0',step_refresh_remainder=16,select_only_in_h=True,runner=run_model_semi_cached_mlp,h=6 || echo "command failed, continuing"


# mlp-1B k=16,h=8
accelerate launch --num_processes=1 run_benchmark_main.py --tasks ifeval --limit 50 --model test --batch_size 1 --num_fewshot 1 --model_args id_model='GSAI-ML/LLaDA-8B-Base',size_batch=1,len_target=256,num_blocks=1,num_unmask_per_step=1,id_mask=126336,device='cuda:0',step_refresh_remainder=16,select_only_in_h=True,runner=run_model_semi_cached_mlp,h=8 || echo "command failed, continuing"


# mlp-1B k=16,h=10
accelerate launch --num_processes=1 run_benchmark_main.py --tasks ifeval --limit 50 --model test --batch_size 1 --num_fewshot 1 --model_args id_model='GSAI-ML/LLaDA-8B-Base',size_batch=1,len_target=256,num_blocks=1,num_unmask_per_step=1,id_mask=126336,device='cuda:0',step_refresh_remainder=16,select_only_in_h=True,runner=run_model_semi_cached_mlp,h=10 || echo "command failed, continuing"


# mlp-1B k=16,h=12
accelerate launch --num_processes=1 run_benchmark_main.py --tasks ifeval --limit 50 --model test --batch_size 1 --num_fewshot 1 --model_args id_model='GSAI-ML/LLaDA-8B-Base',size_batch=1,len_target=256,num_blocks=1,num_unmask_per_step=1,id_mask=126336,device='cuda:0',step_refresh_remainder=16,select_only_in_h=True,runner=run_model_semi_cached_mlp,h=12 || echo "command failed, continuing"


# mlp-1B k=16,h=16
accelerate launch --num_processes=1 run_benchmark_main.py --tasks ifeval --limit 50 --model test --batch_size 1 --num_fewshot 1 --model_args id_model='GSAI-ML/LLaDA-8B-Base',size_batch=1,len_target=256,num_blocks=1,num_unmask_per_step=1,id_mask=126336,device='cuda:0',step_refresh_remainder=16,select_only_in_h=True,runner=run_model_semi_cached_mlp,h=16 || echo "command failed, continuing"


# mlp-1B k=16,h=24
accelerate launch --num_processes=1 run_benchmark_main.py --tasks ifeval --limit 50 --model test --batch_size 1 --num_fewshot 1 --model_args id_model='GSAI-ML/LLaDA-8B-Base',size_batch=1,len_target=256,num_blocks=1,num_unmask_per_step=1,id_mask=126336,device='cuda:0',step_refresh_remainder=16,select_only_in_h=True,runner=run_model_semi_cached_mlp,h=24 || echo "command failed, continuing"
