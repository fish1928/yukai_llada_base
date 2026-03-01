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
from modeling_fastdllm.modeling_llada import LLaDAModelLM

from datetime import datetime, timezone

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
def run_model_with_dual_cache(
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
    is_eval=True
):
    
    x, y = ids_input_masked_full, ids_target_masked_full
    B = x.shape[0]

    probs_all = torch.zeros(x.shape, dtype=torch.float64).to(model.device)

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps_per_block = steps // num_blocks

    nfe = 0

    for id_block in range(num_blocks):
        position_start = len_prompt + id_block * block_length
        position_end = position_start + block_length

        # Masks/indices for the current block
        block_mask_index = (x[:, position_start:position_end] == mask_id)  # (B, block_length)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps_per_block)  # (B, steps_per_block)

        # 1) Warm KV-cache on the full prefix once per block
        out_full = model(x, use_cache=True)
        past_key_values = out_full.past_key_values
        nfe += 1

        # Build a replace_position tensor indicating the block range (static slice)
        mask_position_replace = torch.zeros_like(x, dtype=torch.bool)
        mask_position_replace[:, position_start:position_end] = True  # boolean mask (not a dynamic slice bound)

        # Step 0: do an initial transfer on the full logits
        mask_masked_full = (x == mask_id)
        # Do not touch beyond current block in this phase
        mask_masked_full[:, position_end:] = False

        quota0 = num_transfer_tokens[:, 0]  # (B,)
        x0, x0_p, transfer_index = get_transfer_index(
            out_full.logits,
            temperature,
            remasking,
            mask_masked_full,
            x,
            y,
            quota0
        )

        # In-place update via torch.where (no tensor-slice assignment with mask)
        # x = torch.where(transfer_index, x0, x)   # -> replace by jinyu
        if is_eval:
            x[transfer_index] = y[transfer_index]
            probs_all[transfer_index] = x0_p[transfer_index]
        else:
            x[transfer_index] = x0[transfer_index]
        # end

        # 2) Semi-autoregressive refinement, fixed number of steps (graph-friendly)
        #    Each iteration runs on the current block with KV-cache and replace_position
        for step_in_block in range(1, steps_per_block):
            # Evaluate logits only for current block with cache
            if (x[:, position_start:position_end] == mask_id).sum() == 0:
                break
            # end

            logits_blk = model(
                x[:, position_start:position_end],
                past_key_values=past_key_values,
                use_cache=True,
                replace_position=mask_position_replace
            ).logits  # shape expected by get_transfer_index*

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

            if is_eval:
                blk_x[transfer_idx_blk] = blk_y[transfer_idx_blk]
                blk_prob[transfer_idx_blk] = blk_x0_p[transfer_idx_blk]
            else:
                blk_x[transfer_idx_blk] = blk_x0[transfer_idx_blk]
            # end

            nfe += 1

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

    ds_origin = Dataset.from_list(samples)


    '''initialize constant hyper-parameters'''
    id_model_g = 'GSAI-ML/LLaDA-8B-Base'
    id_mask_g = 126336
    device_g = 'cuda:0'
    size_batch_g = 32


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

    # len_prompt_g = 128
    # len_target_g =  128
    # num_blocks_g = 8
    # num_unmask_per_iter_g = 1

    for len_prompt_g in (64,):
        for len_target_g in (256,):
            for num_blocks_g in (4,):
                for num_unmask_per_iter_g in (1,):

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
                    for batch in tqdm(loader):

                        result = run_model_with_dual_cache(
                            model,
                            batch['ids_prompt_masked_full'].to(device_g),
                            batch['ids_target_masked_full'].to(device_g),
                            len_prompt_g,
                            remasking='truth_top_k',
                            steps=steps_g,
                            gen_length=len_target_g,
                            block_length=size_block_g,
                            mask_id=id_mask_g
                        )

                        with open(f'{len_prompt_g}-{len_target_g}-{num_blocks_g}-{num_unmask_per_iter_g}.dualllada', 'a+') as file:
                            ppl, conf = calculate_ppl_and_conf(result[0], result[1])
                            str_ts_now = get_current_time_str()
                            file.write(f'[{str_ts_now}] {ppl} | {conf}\n')
                        # end
                    # end
                # end
            # end
        # end
    # end
# end if main

