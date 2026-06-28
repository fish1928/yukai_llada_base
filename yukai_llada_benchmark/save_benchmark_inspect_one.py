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
# from run_model_semi import RunModelSemi as RunModel
# from run_model_semi_cached import RunModelSemiCached as RunModel
# from run_model_semi_cached_mlp import RunModelSemiCachedMLP as RunModel
# from run_model_dllm import RunModelDLLM as RunModel



from configs_llada import DiffusionConfig_Eval
from tools_debug import jprint


from constants_llada import DTYPE_EVAL, TEXT_MASK


def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
# end



@register_model("test")
class TestLM(LM):
    def __init__(self, batch_size=1, *args, **kwargs):
        super().__init__()
    # end


    @torch.inference_mode()
    def generate_until(self, requests_eval):    # requests_eval is all

        for req in requests_eval:
            jprint(req)
        # end

        return requests_eval
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