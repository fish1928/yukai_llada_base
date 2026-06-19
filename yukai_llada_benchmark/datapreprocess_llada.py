import torch
from abc import ABC, abstractmethod

import re
from typing import Tuple
from datasets import load_dataset
from collections import defaultdict


PATTEN_REG_WIKI = re.compile(r'^\s*(?P<left>(?:=\s*)+)\s*(?P<text>[^=\n]*?)\s*(?P<right>(?:=\s*)+)\s*$')

def parse_lines_with_index(pat, lines, index=0, target_indent=0) -> tuple[list[str], int]:
    mydoc = {'texts': [], 'subdocs': []}

    while index < len(lines):
        line = lines[index]
        m = pat.match(line)
        if m:
            left_indent = m.group("left").count("=")
            if left_indent < target_indent:
                break   # same return value in exit condition
            elif left_indent == target_indent:
                if len(mydoc['texts']) == 0:
                    mydoc['texts'].append(line.lstrip().rstrip())
                    index += 1
                    continue
                else:   # hit a new-same indent
                    break
                # end
            else: # left_indent > target_indent(cannot be the same)
                subdoc, index = parse_lines_with_index(pat, lines, index, left_indent)
                mydoc['subdocs'].append(subdoc)
            # end
        else:
            if len(line) != 0:
                mydoc['texts'].append(line.lstrip().rstrip())
            # end
            index += 1
            continue
        # end
    # end

    return mydoc, index
# end

def merge_subdocs_deprecated(doc) -> tuple[list[str], list[str]]:
    lines = []
    titles = []

    lines += doc['texts']
    titles.append(subdoc['texts'][0])

    for subdoc in doc['subdocs']:
        sublines, subtitles = merge_subdocs(subdoc)
        lines += sublines
        titles += subtitles
    # end

    return lines, titles
# end

def merge_subdocs(subdocs) -> tuple[list[str], list[str]]:
    lines = []
    titles = []

    for subdoc in subdocs:
        lines += subdoc['texts']
        titles.append(subdoc['texts'][0])

        sublines, subtitles = merge_subdocs(subdoc['subdocs'])
        lines += sublines
        titles += subtitles
    # end for
    
    return lines, titles
# end


def simple_calculate_sim(sample, predict):

    dict_token_count_predict = defaultdict(int)
    tokens_predict = [token for token in predict.split(' ') if len(token) > 2]
    for token_predict in tokens_predict:
        dict_token_count_predict[token_predict] += 1
    # end

    dict_token_count_sample = defaultdict(int)
    tokens_sample = [token for token in sample.split(' ') if len(token) > 2]
    tokens_sample = tokens_sample[:min(len(tokens_predict), len(tokens_sample))]

    for token_sample in tokens_sample:
        dict_token_count_sample[token_sample] += 1
    # end

    count_common = 0

    for token, count_predict in dict_token_count_predict.items():
        count_sample = dict_token_count_sample[token]
        count_common += min(count_predict, count_sample)
    # end

    if sum(dict_token_count_predict.values()):
        return count_common / sum(dict_token_count_predict.values())
    else:
        return 0
    # end
# end



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