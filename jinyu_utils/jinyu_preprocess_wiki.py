import re
from typing import Tuple
from datasets import load_dataset

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
                print(f'handling {index} {left_indent} {target_indent}')
                subdoc, index = parse_lines_with_index(pat, lines, index, left_indent)
                mydoc['subdocs'].append(subdoc)
            # end
        else:
            if len(line) != 0:
                line = line.lstrip().rstrip()
                mydoc['texts'].append()
            # end
            index += 1
            continue
        # end
    # end

    return mydoc, index
# end

def merge_subdocs(doc) -> tuple[list[str], list[str]]:
    lines = []
    titles = []

    lines += doc['texts']
    titles += [doc['texts'][0]]

    for subdoc in doc['subdocs']:
        sublines, subtitles = merge_subdocs(subdoc)
        lines += sublines
        titles += subtitles
    # end

    return lines, titles
# end

if __name__ == '__main__':
    pat = PATTEN_REG_WIKI
    names_dataset = [('Idavidrein/gpqa', 'gpqa_main'), ('Salesforce/wikitext', 'wikitext-2-raw-v1')]
    ds = load_dataset(*names_dataset[1], split='test')['text'][:100]
    mydoc, _ = parse_lines_with_index(pat, ds)
    lines, titles = merge_subdocs(mydoc['subdocs'][0])
    print(titles)
# end