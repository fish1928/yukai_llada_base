from datasets import load_dataset

LIST_DATASET = [
    ('Idavidrein/gpqa', 'gpqa_main'),
    ('Salesforce/wikitext', 'wikitext-2-raw-v1'),
    ('openai/gsm8k', 'main')
]

def jinyu_load_dataset(idx_dataset, split='test'):
    return load_dataset(*LIST_DATASET[idx_dataset], split='test')
# end