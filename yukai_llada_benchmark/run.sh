# common
accelerate launch --num_processes=1 run_benchmark_main.py --tasks gsm8k --limit 1 --model test --batch_size 1 --num_fewshot 5 --model_args id_model='GSAI-ML/LLaDA-8B-Base',size_batch=1,len_target=128,num_blocks=2,num_unmask_per_step=1,id_mask=126336,device='cuda:0',runner=run_model_semi

# dllm
accelerate launch --num_processes=1 run_benchmark_main.py --tasks gsm8k --limit 1 --model test --batch_size 1 --num_fewshot 5 --model_args id_model='GSAI-ML/LLaDA-8B-Base',size_batch=1,len_target=128,num_blocks=1,num_unmask_per_step=1,id_mask=126336,device='cuda:0',runner=run_model_dllm

# mlp
accelerate launch --num_processes=1 run_benchmark_main.py --tasks gsm8k --limit 1 --model test --batch_size 1 --num_fewshot 5 --model_args id_model='GSAI-ML/LLaDA-8B-Base',size_batch=1,len_target=128,num_blocks=2,num_unmask_per_step=1,id_mask=126336,device='cuda:0',step_refresh_remainder=16,select_only_in_h=True,runner=run_model_semi_cached_mlp

# run llada semi
accelerate launch --num_processes=1 run_benchmark_main.py --tasks gsm8k --limit 1 --model test --batch_size 1 --num_fewshot 5 --model_args id_model='GSAI-ML/LLaDA-8B-Base',size_batch=1,len_target=128,num_blocks=2,num_unmask_per_step=1,id_mask=126336,device='cuda:0',runner=run_model_semi

# run llada semi log
accelerate launch --num_processes=1 run_benchmark_main.py --tasks gsm8k --limit 1 --model test --batch_size 1 --num_fewshot 5 --model_args id_model='GSAI-ML/LLaDA-8B-Base',size_batch=1,len_target=128,num_blocks=2,num_unmask_per_step=1,id_mask=126336,device='cuda:0',runner=run_model_semi_with_log
