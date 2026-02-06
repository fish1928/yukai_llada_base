import torch
import torch.nn.functional as F
from jinyu_utils import jinyu_dataset
from jinyu_utils.jinyu_preprocess_wiki import parse_lines_with_index, merge_subdocs, PATTEN_REG_WIKI

from transformers import AutoTokenizer
from datasets import load_dataset, Dataset
from abc import ABC, abstractmethod

from torch.utils.data import DataLoader

from llada_get_loglikelihood import forward_process, get_log_likelihood
from llada_generate import get_num_transfer_tokens, add_gumbel_noise

from tqdm import tqdm
from modeling_llada.modeling_llada import LLaDAModelLM



'''initialize global constants'''

ID_TOKEN_MASK = 126336 # '|mdm_mask|'
ID_TOKEN_PADDING = 126081 # '|endoftext|'
ID_TOKEN_EOT = 126348 # '|eot_id|'












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




def calculate_ppl_and_conf(probs_all, mask_target, eps=1e-12):
    # probs_collected = probs_all[mask_target].reshape(mask_target.shape[0], -1)  # [B, K] ALERT: what it was
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

    # return ppl_per.item(), mean_prob.item()
    return ppl_per.item(), mean_prob.item()
# end


@ torch.no_grad()
def run_model(
        model,
        ids_input_masked_full,
        ids_target_masked_full,
        len_prompt,
        remasking='truth_top_k',
        steps=128,
        gen_length=128,
        block_length=128,
        temperature=0,
        mask_id=126336,
        attention_mask=None,
        logits_eos_inf=False,
        confidence_eos_eot_inf=False,
    ):

    # (batch, full_length)
    x = ids_input_masked_full
    y = ids_target_masked_full

    shape_prompt = (x.shape[0], len_prompt)

    probs_all = torch.zeros(x.shape, dtype=torch.bfloat16).to(model.device)


    if attention_mask is not None:
        attention_mask = torch.cat([attention_mask, torch.ones((shape_prompt[0], gen_length), dtype=attention_mask.dtype, device=model.device)], dim=-1)
    # end

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps_per_block = steps // num_blocks
    for num_block in range(num_blocks):

        mask_block = (x[:, shape_prompt[1] + num_block*block_length : shape_prompt[1]+(num_block + 1)*block_length]) == mask_id

        nums_transfer_tokens = get_num_transfer_tokens(mask_block, steps_per_block)    # [[7,7,6],..] if steps_per_block = 3 and remainder = 2
        # DEBUG: print(nums_transfer_tokens)
        del mask_block

        for step_per_block in range(steps_per_block):   # TODO: 1 -> steps_per_block
        
            logits = model(x, attention_mask=attention_mask).logits

            if logits_eos_inf:
                logits[:, :, ID_TOKEN_PADDING] = -torch.inf
            # end

            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1) # b, l -> [[id0_5, id1_3, ...],..] (this is the index of native max)

            if confidence_eos_eot_inf:
                logits_with_noise[:, :, ID_TOKEN_PADDING] = logits[:, :, ID_TOKEN_EOT] = -torch.inf
            # end

            p = F.softmax(logits, dim=-1)
            del logits

            index_p = None  # we are going to handle the index_p
            match remasking:
                case 'generate_top_k':
                    index_p = x0.unsqueeze(-1)    # ALERT: original code
                    # index_p = y.unsqueeze(-1)
                case 'truth_top_k' | 'random_top_k':
                    index_p = y.unsqueeze(-1)
                case _:
                    raise NotImplementedError(remasking)
                # end
            # end match
            x0_p = torch.squeeze(p.gather(dim=-1, index=index_p), -1) # b, l [[0.9, 0.7],..]

            # set mask
            mask_current_full = torch.where(x==mask_id, True, False)    # set prompt to False
            mask_current_full[:, (shape_prompt[1]+(num_block+1)*block_length):] = False # set future block to False
            # print((mask_current_full == True).sum())  # DEBUG Random issue here
            # print(x0_p[mask_current_full])

            x0 = torch.where(mask_current_full, x0, x)  # restore non-current-block tokens
            x0_p_current_full = torch.where(mask_current_full, x0_p, -torch.inf)

            # update x0, keep x0=x0 for the current & future blocks, override x0=x for prompt
            mask_transfered = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            k = nums_transfer_tokens[:, step_per_block][0]  # WARN: hardcode to get the first k assuming all k's are the same

            if remasking == 'random_top_k':
                idx_mask_current = mask_current_full.nonzero(as_tuple=True)[-1].reshape(mask_current_full.shape[0], -1) # 1d
                perm = torch.argsort(torch.rand(idx_mask_current.shape[0], idx_mask_current.shape[1], device=mask_current_full.device), dim=-1)
                idx_unmask_k = idx_mask_current.gather(-1, perm)[:,:k]
            else:
                _, idx_unmask_k = torch.topk(x0_p_current_full, k)
            # end if-else

            mask_transfered.scatter_(-1, idx_unmask_k, True)            # VALID format of this mask_transfered[idx_unmask_k] = True

            x[mask_transfered] = y[mask_transfered]                     # original code:   x[mask_transfered] = x0[mask_transfered]
            probs_all[mask_transfered] = x0_p_current_full[mask_transfered]   # 
            # end
        # end for steps
    # end for blocks
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


    '''load model tokenizer'''
    tokenizer = AutoTokenizer.from_pretrained(
        id_model_g,
        trust_remote_code=True
    )


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

    for len_prompt_g in (32,64,128):
        for len_target_g in (128, 256, 512):
            for num_blocks_g in (8,4,1):
                for num_unmask_per_iter_g in (1,2,4):

                    '''hyper parameter can be calculated'''
                    len_max_g = len_prompt_g + len_target_g
                    size_block_g = int(len_target_g / num_blocks_g)
                    assert num_unmask_per_iter_g <= size_block_g
                    steps_g = int(len_target_g / num_unmask_per_iter_g)


                    '''start to handle dataset based on hyper-parameter'''
                    ds = ds_origin\
                        .filter(lambda x: x["text"] is not None and len(x["text"].strip()) > 0)\
                        .map(Tokenizer_wiki_simple(tokenizer, len_max_g), remove_columns=ds.column_names)\
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
                        result = run_model(
                            model,
                            batch['ids_prompt_masked_full'].to(device_g),
                            batch['ids_target_masked_full'].to(device_g),
                            len_prompt_g,
                            'truth_top_k',
                            steps=steps_g,
                            gen_length=len_target_g,
                            block_length=size_block_g,
                            temperature=0,
                            mask_id=id_mask_g,
                            attention_mask=None           
                        )

                        with open(f'{len_prompt_g}-{len_target_g}-{num_blocks_g}-{num_unmask_per_iter_g}', 'a+') as file:
                            ppl, conf = calculate_ppl_and_conf(result[0], result[1])
                            file.write(f'{ppl} | {conf}\n')
                        # end
                    # end
                # end
            # end
        # end
    # end
# end if

