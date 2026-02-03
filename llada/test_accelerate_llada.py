import accelerate as ac
import torch
from transformers import AutoModel, AutoTokenizer


### parameters
num_mc = 128
size_batch = 32
eps_sampling = 0.
length_max = 128
cfg = 0.
steps = 128
gen_length = 128
length_block = 32
remasking = 'low_confidence'
# id_model = 'GSAI-ML/LLaDA-8B-Base'
id_model = 'distilbert-base-uncased'


### init accelerator
accelerator = ac.Accelerator()
device = accelerator.device

try:
    ### load model
    model = AutoModel.from_pretrained(id_model, trust_remote_code=True, device_map={'': f'{device}'})
    model = model.eval()
    model = accelerator.prepare(model)
    _rank = accelerator.local_process_index
    _world_size = accelerator.num_processes

    ### load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(id_model, trust_remote_code=True)
    if tokenizer.padding_side != 'left':
        tokenizer.padding_side = 'left'
    # end if 
    assert tokenizer.pad_token_id != 126336

    print(f'rank {_rank} in world {_world_size} with device {device}')
# end


finally:
    accelerator.end_training()
# end