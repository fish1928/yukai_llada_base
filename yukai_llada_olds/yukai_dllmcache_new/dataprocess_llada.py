import torch
from abc import ABC, abstractmethod



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