class LLaDALlamaBlock(LLaDABlock):
    """
    This is a transformer block where the output is computed as ``MLP(LN(x + Attention(LN(x))))``
    (plus another skip connection). This block is similar to `LLaDASequentialBlock`
    but some operations have slightly different implementations to imitate the
    behavior of Llama.
    """

    def __init__(self, layer_id: int, config: ModelConfig, cache: BufferCache):
        super().__init__(layer_id, config, cache)
        # Layer norms.
        self.attn_norm = LayerNorm.build(config)
        self.ff_norm = LayerNorm.build(config)
        self.__cache = cache

        # Attention input projection. Projects x -> (q, k, v)
        head_dim = config.d_model // config.n_heads
        q_proj_out_dim = config.d_model
        k_proj_out_dim = config.effective_n_kv_heads * head_dim
        v_proj_out_dim = config.effective_n_kv_heads * head_dim
        self.q_proj = nn.Linear(
            config.d_model,
            q_proj_out_dim,
            bias=config.include_bias | config.include_qkv_bias,
            device=config.init_device
        )

        self.k_proj = nn.Linear(
            config.d_model,
            k_proj_out_dim,
            bias=config.include_bias | config.include_qkv_bias,
            device=config.init_device
        )

        self.v_proj = nn.Linear(
            config.d_model,
            v_proj_out_dim,
            bias=config.include_bias | config.include_qkv_bias,
            device=config.init_device
        )

        # Feed-forward input projection.
        self.ff_proj = nn.Linear(
            config.d_model,
            self.hidden_size,
            bias=config.include_bias,
            device=config.init_device
        )

        # new add
        self.up_proj = nn.Linear(
            config.d_model,
            self.hidden_size,
            bias=config.include_bias,
            device=config.init_device
        )
    # end

    def reset_parameters(self):
        super().reset_parameters()
        self.attn_norm.reset_parameters()
        self.ff_norm.reset_parameters()
        # NOTE: the standard deviation for these weights does not depend on the layer.
        init_weights(self.config, self.q_proj, d=self.config.d_model, layer_id=None)
        init_weights(self.config, self.k_proj, d=self.config.d_model, layer_id=None)
        init_weights(self.config, self.v_proj, d=self.config.d_model, layer_id=None)
        init_weights(self.config, self.ff_proj, d=self.config.d_model, layer_id=None)
        init_weights(self.config, self.up_proj, d=self.config.d_model, layer_id=None)  # new add
    # end
    

    def forward(
        self,
        x: torch.Tensor,
        attention_bias: Optional[torch.Tensor] = None,
        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        idx_refresh: Optional[torch.Tensor] = None,
        idx_denoising: Optional[torch.Tensor] = None,
        shape_target: Tuple[int, int, int] = None
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        '''
        Get query, key, value projections.
        shape:
            - for regular attn q, k, v: (batch_size, seq_len, d_model)
            - for multi-query attn q: (batch_size, seq_len, d_model)
                                k, v: (batch_size, seq_len, d_model // n_heads)
            - for group query attn q: (batch_size, seq_len, d_model)
                                k, v: (batch_size, seq_len, d_model // n_kv_heads)
            - [current] = [refresh|denoising]
        '''

        x_normed = self.attn_norm(x) #x:torch.Size([2, 168, 4096])
        x_normed_denoising = x_normed[:, -idx_denoising.shape[1], :]
        q_denoising = self.q_proj(x_normed_denoising) #q:torch.Size([2, 168, 4096])

        k = self.k_proj(x_normed) #k:torch.Size([2, 168, 4096])
        v = self.v_proj(x_normed) #v:torch.Size([2, 168, 4096])

        if self._activation_checkpoint_fn is not None:
            att, cache = self._activation_checkpoint_fn(  # type: ignore
                self.attention, q_denoising, k, v, attention_bias=attention_bias,
                layer_past=layer_past,
                use_cache=use_cache,
                idx_refresh=idx_refresh,
                idx_denoising=idx_denoising
            )
        else:
            att, cache = self.attention(
                q_denoising, k, v, attention_bias=attention_bias,
                layer_past=layer_past,
                use_cache=use_cache,
                idx_refresh=idx_refresh,
                idx_denoising=idx_denoising,
                shape_target=shape_target
            )
        # end

        # Add attention scores.
        # shape: (B, T, C)
        x = x + self.dropout(att)

        # Add feed-forward projection.
        # shape: (batch_size, seq_len, d_model)
        og_x = x

        if self._activation_checkpoint_fn is not None:
            x = self._activation_checkpoint_fn(self.ff_norm, x)  # type: ignore
        else:
            x = self.ff_norm(x)
        # end

        x, x_up = self.ff_proj(x), self.up_proj(x) # new add

        if self._activation_checkpoint_fn is not None:
            x = self._activation_checkpoint_fn(self.act, x)  # type: ignore
        else:
            x = self.act(x)
        # end

        x = x * x_up # new add
        x = self.ff_out(x)
        x = self.dropout(x)
        x = og_x + x

        return x, cache
    # end
# end

def attention(
    self,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    attention_bias: Optional[torch.Tensor] = None,
    layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    use_cache: bool = False,
    idx_refresh: Optional[torch.Tensor] = None,
    idx_denoising: Optional[torch.Tensor] = None,
    shape_target=Optional[Tuple[int, int, int]] = None

) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
    
    B, T, C = q.size()  # batch size, sequence length, d_model
    dtype = k.dtype

    # Optionally apply layer norm to keys and queries.
    if self.q_norm is not None and self.k_norm is not None: #self.q_norm: None, self.k_norm: None
        q = self.q_norm(q).to(dtype=dtype)
        k = self.k_norm(k).to(dtype=dtype)

    # Move head forward to be next to the batch dim.
    # shape: (B, nh, T, hs)
    # self.config.n_heads: 32
    q = q.view(B, T, self.config.n_heads, C // self.config.n_heads).transpose(1, 2)
    # shape: (B, n_kv_h, T, hs)
    k = k.view(B, T, self.config.effective_n_kv_heads, C // self.config.n_heads).transpose(1, 2)
    # shape: (B, n_kv_h, T, hs)
    v = v.view(B, T, self.config.effective_n_kv_heads, C // self.config.n_heads).transpose(1, 2)

    if layer_past is not None: 
        past_key, past_value = layer_past
        if replace_position is None:
            k = torch.cat((past_key, k), dim=-2)
            v = torch.cat((past_value, v), dim=-2)
        else:
            # k shape is [B, n_kv_h, selected_length, hs]
            # replace_position shape is [B, L], where L contains 0s and 1s, 0 means no replacement, 1 means replace, with selected_length number of 1s
            # past_key shape is [B, n_kv_h, L, hs]
            # Replace selected_length number of 1s in past_key with k
            
            # Handle batched replace_position correctly
            B = replace_position.shape[0]
            for batch_idx in range(B):
                # Get indices for this batch
                batch_replace_indices = replace_position[batch_idx].nonzero(as_tuple=True)[0]
                if len(batch_replace_indices) > 0:
                    # Replace positions in past_key and past_value for this batch
                    past_key[batch_idx, :, batch_replace_indices] = k[batch_idx, :, :len(batch_replace_indices)]
                    past_value[batch_idx, :, batch_replace_indices] = v[batch_idx, :, :len(batch_replace_indices)]
                # end
            # end
            
            k = past_key
            v = past_value
        # end if replace_position
    # end if layer_past

    present = (k, v) if use_cache else None #present: None
    query_len, key_len = q.shape[-2], k.shape[-2]  # could be different if layer_past not None

    if self.config.rope:
        # Apply rotary embeddings.
        if replace_position is None:
            q, k = self.rotary_emb(q, k)
        else:
            # For batched replace_position, use the maximum position across all batches
            max_replace_pos = replace_position.nonzero(as_tuple=True)[1].max() + 1 if replace_position.any() else key_len
            q, k = self.rotary_emb(q, k, max_replace_pos)

    if attention_bias is not None:
        # Resize and cast attention bias.
        # The current dtype of the attention bias might not match the dtype that the SDP attn function will
        # run in if AMP is enabled, and this can be a problem if some tokens are masked out due to padding
        # as down-casting the attention bias to the autocast precision will result in -infs, which will
        # cause the SDP attn function to produce NaNs.
        attention_bias = self._cast_attn_bias(
            attention_bias[:, :, key_len - query_len : key_len, :key_len], dtype
        )

    # Get the attention scores.
    # shape: (B, nh, T, hs)
    att = self._scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=None,
        dropout_p=0.0 if not self.training else self.config.attention_dropout,
        is_causal=False,
    )
    # Re-assemble all head outputs side-by-side.
    att = att.transpose(1, 2).contiguous().view(B, T, C)

    # Apply output projection.
    return self.attn_out(att), present
# end
