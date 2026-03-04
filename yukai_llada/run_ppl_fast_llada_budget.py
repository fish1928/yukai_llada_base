import os
import torch
import torch.nn.functional as F
from jinyu_utils import jinyu_dataset
from jinyu_utils.jinyu_preprocess_wiki import parse_lines_with_index, merge_subdocs, PATTEN_REG_WIKI

from transformers import AutoTokenizer
from datasets import load_dataset, Dataset
from abc import ABC, abstractmethod

from torch.utils.data import DataLoader

from fastdllm_generate import add_gumbel_noise, get_num_transfer_tokens

from tqdm import tqdm
from modeling_llada_with_budget_and_cachekv import LLaDAModelLM

from datetime import datetime, timezone
from collections import defaultdict

def get_current_time_str():
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
# end



'''initialize global constants'''

ID_TOKEN_MASK = 126336 # '|mdm_mask|'
ID_TOKEN_PADDING = 126081 # '|endoftext|'
ID_TOKEN_EOT = 126348 # '|eot_id|'

TYPES_REMASKING = {'truth_top_k', 'random_top_k'}



'''define token encoder function'''

class Tokenizer_(ABC):
    def __init__(self, tokenizer, len_max):
        self.tokenizer = tokenizer
        self.len_max = len_max
    # end

    @abstractmethod
    def _tokenize(self, ds_each):
        pass
    # end

    def __call__(self, ds_each):
        return self._tokenize(ds_each)
    # end
# end

class Tokenizer_wiki_simple(Tokenizer_):

    def _tokenize(self, ds_each):
        ids = self.tokenizer(
            ds_each['text'],
            add_special_tokens=False,               # avoids BOS/EOS being injected by tokenizer
            truncation=(self.len_max is not None),  # truncation and max_length is a pair
            max_length=self.len_max,
            # return_tensors='pt'
        )["input_ids"]


        return {
            'ids_input': ids,
            'length': len(ids)
        }
    # end tokenize
# end


class Collater_(ABC):
    def __init__(self, len_max, len_prompt, len_target, id_mask):
        self.len_max = len_max
        self.len_prompt = len_prompt
        self.len_target = len_target
        self.id_mask = id_mask
    # end

    @abstractmethod
    def _collate(self, ds_batch):
        pass
    # end

    def __call__(self, ds_batch):
        return self._collate(ds_batch)
    # end
# end


class Collater_wiki_simple(Collater_):

    def _collate(self, ds_batch):
        # batch: list of dicts with "input_ids" as python lists
        len_min = min(len(ds_each["ids_input"]) for ds_each in ds_batch)

        ids_input = torch.stack([torch.tensor(ds_each["ids_input"][:len_min], dtype=torch.long) for ds_each in ds_batch], dim=0) # [B, min_len]
        masks_input = torch.zeros_like(ids_input, dtype=bool)
        masks_input[:, self.len_prompt:] = True
        ids_target = torch.where(masks_input, ids_input, self.id_mask)
        ids_input[masks_input] = self.id_mask

        return {
            'ids_prompt_masked_full': ids_input,
            'ids_target_masked_full': ids_target
        }
    # end _collate
# end

def get_refresh_idx(x, len_prompt=0, type_refresh=None):
    
    if type_refresh == 'previous_all':
        return torch.arange(len_prompt, dtype=torch.long, device=x.device)
    # end

    return torch.tensor([0,1], dtype=torch.long, device=x.device)
# end


''' ppl calculation function'''
def calculate_ppl_and_conf(probs_all, mask_target, eps=1e-12):
    # probs_collected = probs_all[mask_target].reshape(mask_target.shape[0], -1)  # [B, K]
    probs_collected = probs_all[mask_target].reshape(-1)  # [B * K]

    # Arithmetic mean confidence (what you currently call mean_prob)
    mean_prob = probs_collected.mean(dim=-1)  # [B]

    # Per-token NLL and per-row PPL (geometric-mean based)
    nll_collected = -torch.log(probs_collected + eps)   # [B, K]
    nll_per = nll_collected.mean(dim=-1)                 # [B]
    ppl_per = torch.exp(nll_per)                        # [B]

    # Geometric mean confidence (this one is directly tied to PPL)
    # geo_prob = torch.exp(torch.log(probs_collected + eps).mean(dim=1))  # [B]
    # And ppl_per == 1 / geo_prob (up to eps effects)

    return ppl_per.item(), mean_prob.item()
# end


@ torch.no_grad()
def get_transfer_index(
    logits: torch.Tensor,
    temperature: float,
    remasking: str,
    mask_index: torch.Tensor,   # (B, L) bool
    x: torch.Tensor,            # (B, L) long
    y: torch.Tensor,            # (B, L) long
    num_transfer_tokens,        # (B,) or (B,1) long tensor, or None when threshold is used
    threshold: float = None,
):
    """
    Returns:
        x0: (B, L) long — proposed tokens
        transfer_index: (B, L) bool — which positions to update this step
    """
    # 1) Sample proposal x0
    # Gumbel-noise for exploration; if temperature==0, add_gumbel_noise should no-op
    logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
    x0 = torch.argmax(logits_with_noise, dim=-1)  # (B, L), long

    # 2) Confidence for chosen tokens (or random)
    p = F.softmax(logits.to(torch.float64), dim=-1)
    x0_p = torch.gather(p, dim=-1, index=y.unsqueeze(-1)).squeeze(-1)  # (B, L), float64
    # x0_p = torch.rand(x0.shape, device=x0.device, dtype=torch.float64)  # (B, L)  # removed by jinyu

    # Only modify masked spots; keep others as original x and set their confidence to -inf
    # TODO: we have error here
    x0 = torch.where(mask_index, x0, x) # mask_index is only this block

    neg_inf = torch.tensor(torch.finfo(x0_p.dtype).min, device=x0.device, dtype=x0_p.dtype)
    confidence = torch.where(mask_index, x0_p, neg_inf)  # (B, L)   # so only the masked part has confidence

    # Ensure shape (B,) long    jinyu: re-calculate num_transfer_token every time(I think)
    if num_transfer_tokens.dim() == 2 and num_transfer_tokens.size(1) == 1:
        num_transfer_tokens = num_transfer_tokens.squeeze(1)
    # end

    num_transfer_tokens = num_transfer_tokens.to(dtype=torch.long, device=confidence.device)
    num_transfer_tokens = torch.clamp(num_transfer_tokens, min=0)   # jinyu: can it be negative???


    # Sort confidences descending (masked positions are valid; others are -inf)
    # idx: (B, L) gives positions in original sequence sorted by confidence
    if remasking == 'random_top_k':
        idx_sorted_random = torch.argsort(
            torch.where(
                mask_index,
                torch.rand(confidence.shape[0], confidence.shape[1], device=confidence.device),
                confidence
            ),
            dim=1,
            descending=True
        )
        idx_sorted = idx_sorted_random  # for your read
    elif remasking == 'truth_top_k':
        idx_sorted = torch.argsort(confidence, dim=1, descending=True)
    else:
        raise NotImplementedError()
    # end

    B, L = confidence.shape
    # Build a mask that is True for the first k[b] columns in each row (sorted order)
    cols = torch.arange(L, device=confidence.device).unsqueeze(0).expand(B, L)   # (B, L)
    k_expanded = num_transfer_tokens.unsqueeze(1).expand(B, L)                   # (B, L)
    select_sorted = cols < k_expanded                                            # (B, L) bool for top k

    # Scatter the sorted True/False back to original column order
    # Use integer scatter then cast to bool (scatter_ on bool can be finicky across versions)
    transfer_int = torch.zeros(B, L, device=confidence.device, dtype=torch.int8) # (B, L)
    transfer_int = transfer_int.scatter(1, idx_sorted, select_sorted.to(torch.int8))
    transfer_index = transfer_int.bool() & mask_index  # ensure we never select unmasked

    return x0, x0_p, transfer_index
# end





@torch.no_grad()
def run_model_without_budget_and_collect_kv(
    model,
    ids_input_masked_full,
    ids_target_masked_full,
    len_prompt,
    remasking='truth_top_k',
    steps=128,
    gen_length=128,
    block_length=128,
    temperature=0.,
    mask_id=126336,
    is_eval=True,
    cache_kv_previous=True,
    id_batch=0,
    path_output='sims_kv'
):
    
    x, y = ids_input_masked_full, ids_target_masked_full
    B = x.shape[0]

    probs_all = torch.zeros(x.shape, dtype=torch.float64).to(model.device)

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps_per_block = steps // num_blocks

    names_hidden = ('_k_previous','_v_previous')
    dict_hidden_to_matrixs_sim_per_step = {}
    for name_hidden in names_hidden:
        dict_hidden_to_matrixs_sim_per_step[name_hidden] = []
    # end

    dict_cache_kv_previous = {}


    for id_block in range(num_blocks):

        position_start = len_prompt + id_block * block_length
        position_end = position_start + block_length
        block_mask_index = (x[:, position_start:position_end] == mask_id)  # (B, block_length)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps_per_block)  # (B, steps_per_block)

        for step_in_block in range(steps_per_block):
            # Evaluate logits only for current block with cache
            if (x[:, position_start:position_end] == mask_id).sum() == 0:
                break
            # end

            idx_refresh = get_refresh_idx(x, position_start, type_refresh='previous_all') # full prompt is to refresh
            idx_denoising = torch.arange(position_start, position_end,dtype=torch.long, device=x.device)
            idx_current = torch.cat([idx_refresh, idx_denoising])
            x_current = x[:, idx_current]

            output_current = model(
                x_current,
                use_cache=False,
                idx_current=idx_current,
                shape_target=(x.shape[0], position_end, -1),
                cache_kv_previous=cache_kv_previous
            )

            # (B, [refresh|blk])
            logits_current = output_current.logits
            logits_blk = logits_current[:, idx_denoising] # (B, [blk])

            # Mask and quota for this step (all tensor ops)
            mask_blk = (x[:, position_start:position_end] == mask_id)  # (B, block_length)
            blk_x = x[:, position_start:position_end]
            blk_y = y[:, position_start:position_end]
            blk_prob = probs_all[:, position_start:position_end]

            quota_i = num_transfer_tokens[:, step_in_block]  # (B,)
            blk_x0, blk_x0_p, transfer_idx_blk = get_transfer_index(
                logits_blk,
                temperature,
                remasking,
                mask_blk,
                blk_x,
                blk_y,
                quota_i
            )

            blk_x[transfer_idx_blk] = blk_y[transfer_idx_blk] if is_eval else blk_x0[transfer_idx_blk]
            blk_prob[transfer_idx_blk] = blk_x0_p[transfer_idx_blk]

            if not cache_kv_previous:
                continue
            #

            dict_hidden_to_sims_layer = {}
            for name_hidden in names_hidden:
                dict_hidden_to_sims_layer[name_hidden] = []
            # end

            for block_transformer in model.model.transformer.blocks[:]:                       # take last all layers
                id_block_transformer = block_transformer.layer_id
                name_cache_base = f'batch_{id_batch}_layer_{id_block_transformer}'  # block and step in block

                for name_hidden in names_hidden:
                    if hasattr(block_transformer, name_hidden):
                        cache_current = getattr(block_transformer, name_hidden)
                        name_cache = f'{name_cache_base}.{name_hidden}'

                        if name_cache not in dict_cache_kv_previous:
                            dict_cache_kv_previous[name_cache] = cache_current
                            continue
                        # end

                        # we have current and last, calculate similarity
                        cache_last = dict_cache_kv_previous[name_cache]
                        dict_cache_kv_previous[name_cache] = cache_current  # udpate cache

                        if cache_last.shape[1] < cache_current.shape[1]:
                            cache_last = torch.cat([cache_last, cache_current[:, cache_last.shape[1]:, :]], dim=1)
                            # cache_last = F.pad(cache_last, (0,0,0,  cache_current.shape[1] - cache_last.shape[1]), value=0.0)
                            
                        # end

                        sim_neighbour = F.cosine_similarity(cache_current, cache_last, dim=-1)
                        
                        if sim_neighbour.shape[-1] < x.shape[-1]:
                            sim_neighbour_padded = F.pad(
                                sim_neighbour,
                                (0, x.shape[-1]-sim_neighbour.shape[1]),
                                value=1.0
                            ).squeeze(0)
                        else:
                            sim_neighbour_padded = sim_neighbour.squeeze(0)
                        # end

                        dict_hidden_to_sims_layer[name_hidden].append(sim_neighbour_padded)
                    # end
                # end
            # end

            for name_hidden in names_hidden:
                sims_layer = dict_hidden_to_sims_layer[name_hidden]

                if len(sims_layer) == 0:
                    break
                # end
                
                matrix_sim_per_step = torch.stack(sims_layer, dim=0)
                dict_hidden_to_matrixs_sim_per_step[name_hidden].append(matrix_sim_per_step)
            # end

        # end for step
    # end for block

    os.makedirs(path_output, exist_ok=True)

    for name_hidden in names_hidden:
        matrixs_sim_per_step = dict_hidden_to_matrixs_sim_per_step[name_hidden]
        matrix_sim_per_step_layer_token = torch.stack(matrixs_sim_per_step, 0)
        name_sim_final = f'batch_{id_batch}{name_hidden}.pt'
        path_sim_final = os.path.join(path_output, name_sim_final)
        # print(f'saving {path_sim_final} with shape {matrix_sim_per_step_layer_token.shape}')
        torch.save(matrix_sim_per_step_layer_token.detach().float().cpu(), path_sim_final)
    # end


    return probs_all, y != mask_id
# end


if __name__ == '__main__':

    '''load dataset first'''
    name_dataset = jinyu_dataset.LIST_DATASET[1]
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

    print(len(samples))

    ds_origin = Dataset.from_list(samples[:100])


    '''initialize constant hyper-parameters'''
    id_model_g = 'GSAI-ML/LLaDA-8B-Base'
    id_mask_g = 126336
    device_g = 'cuda:0'
    size_batch_g = 1


    '''load tokenizer'''
    tokenizer = AutoTokenizer.from_pretrained(
        id_model_g,
        trust_remote_code=True
    )

    if tokenizer.padding_side != 'left':
        tokenizer.padding_side = 'left'
    # end
    assert tokenizer.pad_token_id != 126336


    '''load model'''
    model_kwargs = {}
    model = LLaDAModelLM.from_pretrained(
        id_model_g,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        **model_kwargs
    )

    model = model.eval().to(device_g)

    '''hyper parameter to be set'''
    len_prompt_g = 128
    len_target_g = 256
    num_blocks_g = 4
    num_unmask_per_iter_g = 1
    path_output_g = 'sims_kv'


    '''hyper parameter can be calculated'''
    len_max_g = len_prompt_g + len_target_g
    size_block_g = int(len_target_g / num_blocks_g)
    assert num_unmask_per_iter_g <= size_block_g
    steps_g = int(len_target_g / num_unmask_per_iter_g)


    '''start to handle dataset based on hyper-parameter'''
    ds = ds_origin\
        .filter(lambda x: x["text"] is not None and len(x["text"].strip()) > 0)\
        .map(Tokenizer_wiki_simple(tokenizer, len_max_g), remove_columns=ds_origin.column_names)\
        .filter(lambda x: x["length"] >= len_max_g)\
        .sort("length")
    # end

    '''prepare dataloader'''
    loader = DataLoader(
        ds,
        batch_size=size_batch_g,
        shuffle=False,                 # keep sorted order
        collate_fn=Collater_wiki_simple(len_max_g, len_prompt_g, len_target_g, id_mask_g),
        drop_last=False
    )


    '''start the evaluation process'''
    for id_batch, batch in enumerate(tqdm(loader)):

        run_model_without_budget_and_collect_kv(
            model,
            batch['ids_prompt_masked_full'].to(device_g),
            batch['ids_target_masked_full'].to(device_g),
            len_prompt_g,
            remasking='truth_top_k',
            steps=steps_g,
            gen_length=len_target_g,
            block_length=size_block_g,
            mask_id=id_mask_g,
            id_batch=id_batch,
            path_output=path_output_g
        )

    # end for
# end