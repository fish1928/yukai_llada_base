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

from tools_llada import TopKSorter, MaxCollector
from modeling_llada_yukai_06 import LLaDAModelLM
from run_model import RunModelSemiCached
from configs_llada import DiffusionConfig_Eval
from tools_debug import jprint


DTYPE_EVAL = torch.bfloat16
MASK_TEXT = '<|mdm_mask|>'

def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



'''define token encoder function'''
class Tokenizer_(ABC):

    def __init__(self, tokenizer, config):
        self.tokenizer = tokenizer
        self.config = config
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
            'id_mask': self.config.id_mask
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
            ds_batch = ds_batch[0]  #<- hit
        # end

        id_mask = ds_batch['id_mask']
        ids_prompt = ds_batch['ids_prompt']
        len_prompt = len(ids_prompt)

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

        self.config = DiffusionConfig_Eval(
            id_model=kwargs['id_model'],
            len_target=kwargs['len_target'],
            num_blocks=kwargs['num_blocks'],
            num_unmask_per_step=kwargs['num_unmask_per_step'],
            id_mask=kwargs['id_mask'],
            size_batch=kwargs['size_batch'],
            device=kwargs['device'],
            klass_sorter=TopKSorter,
            klass_collector=MaxCollector
        )

        self.tokenizer = self._init_tokenizer(self.config.id_model)
        self.model = self._init_model(self.config.id_model).eval().to(self.config.device)
        self.runner_model = RunModelSemiCached()

        self.runner_model.config_plugin_(self.config)
        self.runner_model.register_plugin_(self.model, self.config)
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
    def generate_until(self, requests_eval):    # requests_eval is all
        requests_eval = requests_eval[:1]

        ds = [{"prompt": req_eval.args[0], "until": req_eval.args[1]['until']} for req_eval in requests_eval]
        ds = Dataset.from_list(ds)
        ds = ds.map(Tokenizer_Until(self.tokenizer))

        '''prepare dataloader'''
        loader = DataLoader(
            ds,
            batch_size=self.size_batch,
            shuffle=False,
            drop_last=False,
            collate_fn=Collater_Until_One(self.len_target)
        )

        
        for id_batch, batch in enumerate(tqdm(loader)):
            text_generated, has_done = self.runner_model.run(self.model, self.config, batch)
        # end
    # end


    @torch.inference_mode()
    def loglikelihood_rolling(self, requests):
        raise NotImplementedError
    # end


    @torch.inference_mode()
    def loglikelihood(self, requests):
        raise NotImplementedError
    # end

# end

if __name__ == "__main__":
    set_seed(233)
    cli_evaluate()
# end