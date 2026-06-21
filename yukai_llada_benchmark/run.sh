accelerate launch --num_processes=1 run_tokenizer_test.py --tasks gsm8k --model test --batch_size 1 --num_fewshot 1 --model_args id_model='GSAI-ML/LLaDA-8B-Base',size_batch=1,len_target=128
