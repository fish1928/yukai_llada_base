import importlib
from abc import ABC, abstractmethod

import torch, random
import torch.nn.functional as F
import numpy as np
# import accelerate

from transformers import AutoTokenizer

from datasets import Dataset
from torch.utils.data import DataLoader

from lm_eval.__main__ import cli_evaluate
from lm_eval.api.instance import Instance
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
from tqdm import tqdm

from modeling_llada_yukai_06 import LLaDAModelLM

from tools_debug import jprint


DTYPE_EVAL = torch.bfloat16
MASK_ID = 126336
MASK_TEXT = '<|mdm_mask|>'

def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



'''define token encoder function'''
class Tokenizer_(ABC):

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    # end

    @abstractmethod
    def _tokenize(self, ds_each):
        pass
    # end

    def __call__(self, ds_each):
        return self._tokenize(ds_each)
    # end
# end

class Tokenizer_Until(Tokenizer_):

    def _tokenize(self, ds_each):
        ids = self.tokenizer(
            ds_each['prompt'],
            add_special_tokens=False
        )["input_ids"]


        return {
            'ids_prompt': ids,
            'text_prompt': ds_each['prompt'],
            'until': ds_each['until'],
            'id_mask': MASK_ID
        }
    # end tokenize
# end

class Collater_(ABC):
    @abstractmethod
    def _collate(self, ds_batch):
        pass
    # end

    def __call__(self, ds_batch):
        return self._collate(ds_batch)
    # end
# end

class Collater_Until_One(Collater_):

    def __init__(self, len_target=None):
        self.len_target = len_target
    # end

    def _collate(self, ds_batch):
        if type(ds_batch) is list:
            jprint('is list')
            ds_batch = ds_batch[0]
        # end

        id_mask = ds_batch['id_mask']
        ids_prompt = ds_batch['ids_prompt']
        len_prompt = len(ids_prompt)

        jprint(id_mask, ids_prompt)

        ids_input = ids_prompt + [id_mask] * self.len_target
        ids_input = torch.tensor(ids_input, dtype=torch.long).view(1, -1)
        masks_input = torch.zeros_like(ids_input, dtype=torch.bool)
        masks_input[:, len_prompt:] = True

        return {
            'ids_input': ids_input,
            'masks_input': masks_input,
            'id_mask': id_mask,
            'until': ds_batch['until'],
            'len_target': self.len_target
        }

    # end
# end


@register_model("test")
class TestLM(LM):
    def __init__(self, *args, **kwargs):
        super().__init__()

        id_model = kwargs['id_model']
        self.size_batch = kwargs['size_batch']
        self.len_target = kwargs['len_target']

        self.tokenizer = self._init_tokenizer(id_model)

    # end

    def _init_tokenizer(self, id_model):
        tokenizer = AutoTokenizer.from_pretrained(
            id_model,
            trust_remote_code=True
        )

        if tokenizer.padding_side != 'left':
            tokenizer.padding_side = 'left'
        # end

        assert tokenizer.pad_token_id != 126336
        return tokenizer
    # end

    def _init_model(self, id_model):
        model = LLaDAModelLM.from_pretrained(
            id_model,
            trust_remote_code=True,
            torch_dtype=DTYPE_EVAL,
        )

        return model
    # end


    @torch.inference_mode()
    def loglikelihood(self, requests):
        for request_eval in requests:
            jprint(request_eval)
            raise "loglikelihood"
        # end
    # end


    def generate_until(self, requests_eval):    # requests_eval is all
        requests_eval = requests_eval[:1]

        ds = [{"prompt": req_eval.args[0], "until": req_eval.args[1]['until']} for req_eval in requests_eval]
        ds = Dataset.from_list(ds)
        ds = ds.map(Tokenizer_Until(self.tokenizer))

        jprint(ds[0])

        '''prepare dataloader'''
        loader = DataLoader(
            ds,
            batch_size=self.size_batch,
            shuffle=False,
            drop_last=False,
            collate_fn=Collater_Until_One(self.len_target)
        )
        
        for id_batch, batch in enumerate(tqdm(loader)):
            jprint(batch)
            raise ''
        # end
    # end


    def loglikelihood_rolling(self, requests):
        for request_eval in requests:
            jprint(request_eval)
            raise "loglikelyhood_rolling"
        # end
    # end

# end

if __name__ == "__main__":
    set_seed(233)
    cli_evaluate()
# end
