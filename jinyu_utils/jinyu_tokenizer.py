from abc import ABC, abstractmethod

class Tokenizer_(ABC):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    # end

    def _encode_pair(self, context, continuation):
        n_spaces = len(context) - len(context.rstrip())
        if n_spaces >0:
            continuation = context[-n_spaces:] + continuation
            context = context[:-n_spaces]
        # end

        whole_enc = self.tokenizer(context + continuation)["input_ids"]
        context_enc = self.tokenizer(context)["input_ids"]

        context_enc_len = len(context_enc)
        continuation_enc = whole_enc[context_enc_len:]

        return context_enc, continuation_enc
    # end

    # def _encode_all(self, prefixs):
    #     encoded_outputs = self.tokenizer(
    #         prefixs,
    #         add_special_tokens=False,
    #         padding=True,
    #         return_tensors="pt"
    #     )
    #     input_ids = encoded_outputs['input_ids']
    #     attention_mask = encoded_outputs['attention_mask']

    #     return input_ids, attention_mask
    # # end

    @abstractmethod
    def _tokenize(self, e):
        pass
    # end

    def __call__(self, e):
        return self._tokenize(e)
    # end
# end
