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

    return count_common / sum(dict_token_count_predict.values())
# end

if __name__ == '__main__':
    pat = PATTEN_REG_WIKI
    names_dataset = [('Idavidrein/gpqa', 'gpqa_main'), ('Salesforce/wikitext', 'wikitext-2-raw-v1')]
    ds = load_dataset(*names_dataset[1], split='test')['text'][:100]
    mydoc, _ = parse_lines_with_index(pat, ds)
    lines, titles = merge_subdocs(mydoc['subdocs'][0])
    print(titles)
# end