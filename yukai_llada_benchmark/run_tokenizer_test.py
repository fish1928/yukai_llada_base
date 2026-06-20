import importlib

import torch, random
import torch.nn.functional as F
import numpy as np
# import accelerate

from datasets import Dataset

from lm_eval.__main__ import cli_evaluate
from lm_eval.api.instance import Instance
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
from tqdm import tqdm

from tools_debug import jprint

def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@register_model("test")
class TestLM(LM):
    def __init__(self, *args, **kwargs):
        super().__init__()
        jprint(kwargs)
    # end

    @torch.inference_mode()
    def loglikelihood(self, requests):
        for request_eval in requests:
            jprint(request_eval)
            raise "loglikelihood"
        # end
    # end

    def generate_until(self, requests):
        for request_eval in requests:
            jprint(request_eval)
            raise "generate_until"
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
