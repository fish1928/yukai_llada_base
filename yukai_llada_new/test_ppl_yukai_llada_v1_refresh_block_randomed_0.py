import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # put this at the very top of your script


from dataclasses import dataclass
from typing import Optional
import torch
import torch.nn.functional as F

from transformers import AutoTokenizer
from torch.utils.data import DataLoader

from tools_llada import TopKSorter, TruthCollector, MaxCollector
from plugins_llada import SaveKVPreviousPlugin_Disabled, SaveKVPreviousPlugin_Enabled, CachePastKVPlugin_Disabled, CachePastKVPlugin_Enabled
from datasets import load_dataset, Dataset

from tools_llada import ConfKSorter, ConfCollectorInterface, BlockDiffusionQuotaHelper
from plugins_llada import InspectorPlugin

from dataset_llada import LIST_DATASET
from datapreprocess_llada import parse_lines_with_index, merge_subdocs, PATTEN_REG_WIKI
from dataprocess_llada import Tokenizer_wiki_simple, Collater_wiki_simple

from modeling_yukai_llada import LLaDAModelLM

from tools_debug import jprint

@dataclass
class DiffusionConfig:
    id_model: str
    len_prompt: int
    len_target: int
    num_blocks: int
    num_unmask_per_step: int
    id_mask: int
    size_batch: int
    device: str
    klass_sorter: ConfKSorter
    klass_collector: ConfCollectorInterface
    klass_save_kv_previous: InspectorPlugin
    klass_cache_past_kv: InspectorPlugin
    
    size_block: Optional[int] = None
    step_per_block: Optional[int] = None
# end

@dataclass
class KVAggregateConfig:
    stamp: str
    type_aggregate: str
    len_prompt: str
    len_target: str
    num_blocks: int
    folder_output: Optional[str] = None
    type_fn: Optional[str] = None
# end


config = DiffusionConfig(
    id_model='GSAI-ML/LLaDA-8B-Base',
    len_prompt=128,
    len_target=256,
    num_blocks=4,
    num_unmask_per_step=1,
    id_mask=126336,
    size_batch=1,
    device='cuda:0',
    klass_sorter=TopKSorter,
    klass_collector=TruthCollector,
    klass_save_kv_previous=SaveKVPreviousPlugin_Disabled,
    klass_cache_past_kv=CachePastKVPlugin_Enabled
)

config.size_block= int(config.len_target / config.num_blocks)
config.step_per_block=int(config.size_block / config.num_unmask_per_step)


config_aggregate = KVAggregateConfig(
    stamp='0326',
    type_aggregate='step',
    len_prompt=config.len_prompt,
    len_target=config.len_target,
    num_blocks=config.num_blocks,
    type_fn='p'
)
config_aggregate.folder_output=f'sims_kv_{config_aggregate.stamp}'


'''load dataset first'''
name_dataset = LIST_DATASET[1]
ds = load_dataset(*name_dataset, split='all')
docs, _ = parse_lines_with_index(PATTEN_REG_WIKI, ds['text'])
docs = docs['subdocs']

samples = []
for doc in docs:
    lines_1 = doc['texts']
    paragraph_1 = ' '.join(lines_1)
    lines_remain, titles = merge_subdocs(doc['subdocs'])
    paragraph_remain = ' '.join(lines_remain)
    prefix = paragraph_1
    target = paragraph_remain
    samples.append({'text': paragraph_1 + ' ' + paragraph_remain})
# end

ds_origin = Dataset.from_list(samples[:100])


'''initialize constant hyper-parameters'''

'''load tokenizer'''
tokenizer = AutoTokenizer.from_pretrained(
    config.id_model,
    trust_remote_code=True
)

if tokenizer.padding_side != 'left':
    tokenizer.padding_side = 'left'
# end
assert tokenizer.pad_token_id != 126336


'''load model'''
model_kwargs = {}
model = LLaDAModelLM.from_pretrained(
    config.id_model,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    **model_kwargs
)

model = model.eval().to(config.device)


'''start to handle dataset based on hyper-parameter'''
len_max = config.len_prompt + config.len_target
ds = ds_origin\
    .filter(lambda x: x["text"] is not None and len(x["text"].strip()) > 0)\
    .map(Tokenizer_wiki_simple(tokenizer, len_max), remove_columns=ds_origin.column_names)\
    .filter(lambda x: x["length"] >= len_max)\
    .sort("length")
# end

'''prepare dataloader'''
loader = DataLoader(
    ds,
    batch_size=config.size_batch,
    shuffle=False,                 # keep sorted order
    collate_fn=Collater_wiki_simple(len_max, config.len_prompt, config.len_target, config.id_mask),
    drop_last=False
)

class SimpleLogitsSnapshot:

    def _regularize(self, sample, target):
        return  sample[:, :target.shape[1]]
    # end

    def __init__(self, logits, x, y, id_mask):
        self.id_mask = id_mask

        self.logits = logits

        self.x = self._regularize(x, logits)
        self.y = self._regularize(y, logits)

        self.x0 = torch.argmax(self.logits, dim=-1)

        self.p_finalized = torch.zeros(self.x.shape, dtype=torch.float64).to(self.x.device)
    # end

    def get_x(self):
        return self.x
    # end

    def get_y(self):
        return self.y
    # end

    def get_logits(self):
        return self.logits
    # end

    def get_p_finalized(self):
        return self.p_finalized
    # end

    def transform_logits(self, collector):

        logits_tranform = self.logits
        p = F.softmax(logits_tranform.to(torch.float64), dim=-1)

        index_p_all = collector.get_index(self)

        x0_p = torch.gather(p, dim=-1, index=index_p_all).squeeze(-1)

        neg_inf = torch.tensor(torch.finfo(x0_p.dtype).min, device=x0_p.device, dtype=x0_p.dtype)

        mask_mask = self.x == self.id_mask
        conf = torch.where(mask_mask, x0_p, neg_inf)  # (B, L)   # so only the masked part has confidence

        return conf
    # end

    def materialize_by_idx_(self, idx, conf):

        x0_target = torch.gather(self.x0, dim=-1, index=idx)
        conf_target = torch.gather(conf, dim=-1, index=idx)
        self.x.scatter_(1, idx, x0_target)
        self.p_finalized.scatter_(1, idx, conf_target)
    # end

    def update_logits_(self, idx_transform, logits):
        B, L, H = logits.shape
        assert idx_transform.dim() == 2, "idx_transform.dim(): {} == 2 false".format(idx_transform.dim())
        
        idx_logits = idx_transform.view(B,-1,1).expand(B, -1, H)

        # end match

        self.logits.scatter_(1, idx_logits, logits)
        x0 = torch.argmax(logits, dim=-1)
        self.x0.scatter_(1, idx_transform, x0)
    # end

    def update_this(self, dim, idx_src, idx_tgt=None, **kwargs):

        if idx_tgt is None:
            idx_transform = idx_src
        else:
            idx_tgt=idx_tgt.unsqueeze(0)
            
            idx_transform = torch.gather(idx_tgt, dim=-1, index=idx_src)
        # end

        for k, v in kwargs.items(): # k is a local property name, v is the target to scatter
            v.scatter_(dim, idx_transform, torch.gather(getattr(self, k), dim=dim, index=idx_src))
        # end

        return self
    # end

# end

@ torch.no_grad()
def run_model_semi_cached_refresh(model, x, y, config_diffusion, *args, **kwargs):

    '''declare required variables'''
    num_blocks = config_diffusion.num_blocks
    step_per_block = config_diffusion.step_per_block
    size_block = config_diffusion.size_block
    id_mask = config_diffusion.id_mask
    len_prompt = config_diffusion.len_prompt
    sorter = config_diffusion.klass_sorter()
    collector = config_diffusion.klass_collector()
    refresher = kwargs['refresher']

    p_finalized = torch.zeros(x.shape, dtype=torch.float64, device=x.device)
    idx_denoising = torch.arange(0, len_prompt, dtype=torch.long).to(x.device)
    model(x[:, idx_denoising], idx_current=idx_denoising)   # save prompt for once

    for id_block in range(num_blocks):
        position_start = len_prompt + id_block * size_block
        position_end = position_start + size_block
        mask_mask_blk = x[:,position_start:position_end] == id_mask
        idx_denoising = torch.arange(position_start, position_end, dtype=torch.long).to(x.device)
        quota_helper = BlockDiffusionQuotaHelper(mask_mask_blk, size_block)
        idx_refresh = refresher.get_refresh_idx(x, 0, id_block, return_sorted=True) # 4.87 if idx_refresh get not block, but first step

        for step in range(step_per_block):

            # idx_refresh = refresher.get_refresh_idx(x, step, id_block, return_sorted=True)
            idx_current = torch.cat([idx_refresh, idx_denoising])
            shape_target = (x.shape[0], position_end, -1)
            x_current, x_denoising,  y_denoising= x[:, idx_current], x[:, idx_denoising], y[:, idx_denoising]

            logits_current = model(x_current, idx_current=idx_current, shape_target=shape_target).logits
            logits_denoising = logits_current[:, -idx_denoising.shape[-1]:]
            snapshot = SimpleLogitsSnapshot(logits_denoising, x_denoising, y_denoising, id_mask)
            
            conf_snapshot = snapshot.transform_logits(collector)
            idx_sorted_by_conf = sorter.argsort(conf_snapshot, snapshot)
            num_unmask = quota_helper.get_quota(step)
            idx_transform = idx_sorted_by_conf[:, :num_unmask]

            snapshot.materialize_by_idx_(idx_transform, conf_snapshot)
            snapshot.update_this(1, idx_src=idx_transform, idx_tgt=idx_denoising, y=x).update_this(1, idx_src=idx_transform, idx_tgt=idx_denoising, p_finalized=p_finalized)
        # end for step
    # end for block

    return p_finalized[:, len_prompt:]
# end

import json
from tqdm import tqdm
from tools_llada import PPLCalculator, RefreshIdxHelper

filename = 'all_in_one_diff_128_256_4_abs_per_block_p_0326.json'
with open(filename, 'r') as file:
    data_refresh = json.load(file)
# end

refresher = RefreshIdxHelper(
    data_refresh,
    'v',
    config.size_block,
    randomed=True
)

calculator_ppl = PPLCalculator()
model.fill_plugin(config.klass_cache_past_kv).fill_plugin(config.klass_save_kv_previous)
plugin_cache_past_kv = config.klass_cache_past_kv()

folder_result_base = 'refresh_budget_base'
os.makedirs(folder_result_base, exist_ok=True)

for i in range(10):
    for budget in (1,2,4,8,16,24,32,48):

        '''start the evaluation process'''
        for id_batch, batch in enumerate(tqdm(loader)):
            plugin_cache_past_kv.clear(model)
            refresher.set_sample_id(id_batch).set_budget(budget)

            conf = run_model_semi_cached_refresh(
                model,
                batch['ids_prompt_masked_full'].to(config.device),
                batch['ids_target_masked_full'].to(config.device),
                config,
                refresher=refresher
            )

            type_diffusion = filename.split('per_')[-1].split('_')[0]
            filename_result = f'{folder_result_base}/refresh_budget_{budget}_{type_diffusion}_rand_{i}.txt'
            with open(filename_result, 'a+') as file:
                file.write(f'{calculator_ppl.cal(conf)}\n')
            # end with
        # end for
    # end
# end