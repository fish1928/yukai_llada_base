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
    probs_collected = probs_all[mask_target].reshape(mask_target.shape[0], -1)  # [B, K]

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
# end def


@ torch.no_grad()
def run_model_with_prefix_cache(
    model,
    ids_input_masked_full,
    ids_target_masked_full,
    len_prompt,
    steps=128,
    gen_length=128,
    block_length=128,
    temperature=0.,
    remasking='truth_top_k',
    mask_id=126336,
    is_eval=True
):

    x, y = ids_input_masked_full, ids_target_masked_full
    shape_prompt = (x.shape[0], len_prompt)

    probs_all = torch.zeros(x.shape, dtype=torch.float64).to(model.device)

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps = steps // num_blocks

    nfe = 0
    for num_block in range(num_blocks):
        current_block_start = shape_prompt[1] + num_block * block_length
        current_block_end = current_block_start + block_length

        block_mask_index = (x[:, current_block_start:current_block_end] == mask_id) # block_mask_index is mask_block
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)

        output = model(x, use_cache=True)
        past_key_values = output.past_key_values

        mask_index = (x == mask_id)
        mask_index[:, current_block_end:] = 0

        x0, x0_p, transfer_index = get_transfer_index(
            output.logits,
            temperature,
            remasking,
            mask_index,
            x,
            y,
            num_transfer_tokens[:, 0]
        )

        if is_eval:
            x[transfer_index] = y[transfer_index]
            probs_all[transfer_index] = x0_p[transfer_index]
        else:
            x[transfer_index] = x0[transfer_index]
        # end
        

        new_past_key_values = []
        for i in range(len(past_key_values)):
            new_past_key_values.append(())
            for j in range(len(past_key_values[i])):
                new_past_key_values[i] += (past_key_values[i][j][:, :, :current_block_start],)
            # end for j
        # end for i
        
        past_key_values = new_past_key_values
        nfe += 1

        i = 1
        while True:
            if (x[:, current_block_start:current_block_end] == mask_id).sum() == 0:
                break
                # end
            nfe += 1
            mask_index = (x[:, current_block_start:] == mask_id)
            mask_index[:, block_length:] = 0

            # jinyu: assuming logits is L - len(cached)
            logits = model(x[:, current_block_start:], past_key_values=past_key_values, use_cache=True).logits

            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1) # b, l

            x0, x0_p, transfer_index = get_transfer_index(
                logits,
                temperature,
                remasking,
                mask_index, 
                x[:, current_block_start:],
                y[:, current_block_start:],
                num_transfer_tokens[:, i]
            )

            if is_eval:
                x[:, current_block_start:][transfer_index] = y[:, current_block_start:][transfer_index]
                probs_all[:, current_block_start:][transfer_index] = x0_p[transfer_index]
            else:
                x[:, current_block_start:][transfer_index] = x0[transfer_index]
            # end

            i += 1
        # end for block

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
    device_g = 'cuda:1'
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

    len_prompt_g = 128
    len_target_g =  256
    num_blocks_g = 8
    num_unmask_per_iter_g = 1

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

        result = run_model_with_prefix_cache(
            model,
            batch['ids_prompt_masked_full'].to(device_g),
            batch['ids_target_masked_full'].to(device_g),
            len_prompt_g,
            'truth_top_k',
            steps=steps_g,
            gen_length=len_target_g,
            block_length=size_block_g,
            mask_id=id_mask_g
        )

        with open(f'{len_prompt_g}-{len_target_g}-{num_blocks_g}-{num_unmask_per_iter_g}.fastllada', 'a+') as file:
            ppl, conf = calculate_ppl_and_conf(result[0], result[1])
            str_ts_now = get_current_time_str()
            file.write(f'[{str_ts_now}] {ppl} | {conf}\n')
        # end
    # end





# end if main

