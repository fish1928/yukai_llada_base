import inspect

def jinyu_inspect_file(func_target):
    path_file_func = inspect.getsourcefile(func_target)
    print('start to inspect: {}\n\n'.format(path_file_func))

    with open(path_file_func, 'r') as file:
        print(file.read())
    # end with
# end


id_token_mask = 126336 # '|mdm_mask|'
id_token_padding = 126081 # '|endoftext|'
id_token_eot = 126348 # '|eot_id|'


# @torch.compile()
# def scaled_dot_product_attention(q, k, v, mask=None, attn_mask=None, dropout_p=0.0, is_causal=False):
#     return F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal)


# def _scaled_dot_product_attention_jinyu(
#     self,
#     q: torch.Tensor,
#     k: torch.Tensor,
#     v: torch.Tensor,
#     attn_mask: Optional[torch.Tensor] = None,
#     dropout_p: float = 0.0,
#     is_causal: bool = False,
# ) -> torch.Tensor:
#     """
#     Computes scaled dot product attention on query, key and value tensors, using an optional
#     attention mask if passed, and applying dropout if a probability greater than 0.0 is specified.
#     """
#     if self.flash_attn_func is not None and attn_mask is None:
#         r = self.flash_attn_func(
#             q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), dropout_p=dropout_p, causal=False
#         )
#         return r.transpose(1, 2)
#     else:
#         # torch's sdpa doesn't support GQA, so we're doing this
#         assert k.size(1) == v.size(1)
#         num_kv_heads = k.size(1)
#         num_q_heads = q.size(1)
#         if num_q_heads != num_kv_heads:
#             assert num_q_heads % num_kv_heads == 0
#             k = k.repeat_interleave(num_q_heads // num_kv_heads, dim=1, output_size=num_q_heads)
#             v = v.repeat_interleave(num_q_heads // num_kv_heads, dim=1, output_size=num_q_heads)
#         # end if
        
#         if not hasattr(self, 'jinyu_debug'):
#             self.jinyu_debug = 1
#             print('JINYU DEBUG: attn_mask={}'.format(attn_mask))
#         # end

#         # Modify: MDM set causal to False, and with no attn_mask.
#         return scaled_dot_product_attention(
#             q,
#             k,
#             v,
#             attn_mask=attn_mask,
#             dropout_p=dropout_p,
#             is_causal=False,
#         )
#     # end if-else
# # end def

# model.model.transformer.blocks[0]._scaled_dot_product_attention = _scaled_dot_product_attention_jinyu.__get__(model.model.transformer.blocks[0], )


# def attention_jinyu(
#     self,
#     q: torch.Tensor,
#     k: torch.Tensor,
#     v: torch.Tensor,
#     mask: Optional[torch.Tensor] = None,
#     attention_bias: Optional[torch.Tensor] = None,
#     layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
#     use_cache: bool = False,
#     replace_position: Optional[torch.Tensor] = None,
# ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
#     B, T, C = q.size()  # batch size, sequence length, d_model
#     dtype = k.dtype

#     # Optionally apply layer norm to keys and queries.
#     if self.q_norm is not None and self.k_norm is not None: #self.q_norm: None, self.k_norm: None
#         q = self.q_norm(q).to(dtype=dtype)
#         k = self.k_norm(k).to(dtype=dtype)

#     # Move head forward to be next to the batch dim.
#     # shape: (B, nh, T, hs)
#     # self.config.n_heads: 32
#     q = q.view(B, T, self.config.n_heads, C // self.config.n_heads).transpose(1, 2)
#     # shape: (B, n_kv_h, T, hs)
#     k = k.view(B, T, self.config.effective_n_kv_heads, C // self.config.n_heads).transpose(1, 2)
#     # shape: (B, n_kv_h, T, hs)
#     v = v.view(B, T, self.config.effective_n_kv_heads, C // self.config.n_heads).transpose(1, 2)

#     if layer_past is not None:
#         past_key, past_value = layer_past

#         if replace_position is None:
#             print('jinyuj k: {}, past_key: {}'.format(k.shape, past_key.shape))
#             k = torch.cat((past_key, k), dim=-2)
#             v = torch.cat((past_value, v), dim=-2)
#         else:
#             # k shape is [B, n_kv_h, selected_length, hs]
#             # replace_position shape is [B, L], where L contains 0s and 1s, 0 means no replacement, 1 means replace, with selected_length number of 1s
#             # past_key shape is [B, n_kv_h, L, hs]
#             # Replace selected_length number of 1s in past_key with k

#             # Handle batched replace_position correctly
#             B = replace_position.shape[0]
#             for batch_idx in range(B):
#                 # Get indices for this batch
#                 batch_replace_indices = replace_position[batch_idx].nonzero(as_tuple=True)[0]
#                 if len(batch_replace_indices) > 0:
#                     # Replace positions in past_key and past_value for this batch
#                     past_key[batch_idx, :, batch_replace_indices] = k[batch_idx, :, :len(batch_replace_indices)]
#                     past_value[batch_idx, :, batch_replace_indices] = v[batch_idx, :, :len(batch_replace_indices)]
#                 # end
#             # end

#             k = past_key
#             v = past_value
#         # end else if replace position
#     # end if layer_past

#     present = (k, v) if use_cache else None #present: None
#     query_len, key_len = q.shape[-2], k.shape[-2]  # could be different if layer_past not None

#     if self.config.rope:
#         # Apply rotary embeddings.
#         if replace_position is None:
#             q, k = self.rotary_emb(q, k)
#         else:
#             # For batched replace_position, use the maximum position across all batches
#             max_replace_pos = replace_position.nonzero(as_tuple=True)[1].max() + 1 if replace_position.any() else key_len
#             q, k = self.rotary_emb(q, k, max_replace_pos)
#         # end
#     # end

#     if attention_bias is not None:
#         # Resize and cast attention bias.
#         # The current dtype of the attention bias might not match the dtype that the SDP attn function will
#         # run in if AMP is enabled, and this can be a problem if some tokens are masked out due to padding
#         # as down-casting the attention bias to the autocast precision will result in -infs, which will
#         # cause the SDP attn function to produce NaNs.
#         attention_bias = self._cast_attn_bias(
#             attention_bias[:, :, key_len - query_len : key_len, :key_len], dtype
#         )

#     # Get the attention scores.
#     # shape: (B, nh, T, hs)
    
#     att = self._scaled_dot_product_attention(
#         q,
#         k,
#         v,
#         attn_mask=None,
#         dropout_p=0.0 if not self.training else self.config.attention_dropout,
#         is_causal=False,
#     )
#     # Re-assemble all head outputs side-by-side.
#     att = att.transpose(1, 2).contiguous().view(B, T, C)

#     # Apply output projection.
#     return self.attn_out(att), present

# #a.hello = helloB.__get__(a, A)
# model.model.transformer.blocks[0].attention = attention_jinyu.__get__(model.model.transformer.blocks[0], LLaDALlamaBlock)


def question_and_answer(model, tokenizer, gen_length=128, steps=128, block_size=32):
    user_input = "What is the full name of the U.S."
    m = [{"role": "user", "content": user_input}]
    user_input = tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)
    input_ids = tokenizer(user_input)['input_ids']
    input_ids = torch.tensor(input_ids).to(device).unsqueeze(0)
    prompt = input_ids
    
    # out, nfe = generate(model, prompt, steps=steps, gen_length=gen_length, block_length=block_size, temperature=0., remasking='low_confidence')
    out, nfe = generate_with_prefix_cache(model, prompt, steps=steps, gen_length=gen_length, block_length=block_size, temperature=0., remasking='low_confidence')
    
    answer = tokenizer.batch_decode(out[:, prompt.shape[1]:], skip_special_tokens=True)[0]
    print(f"Bot's reply: {answer}")
# end


question_and_answer(model, tokenizer)