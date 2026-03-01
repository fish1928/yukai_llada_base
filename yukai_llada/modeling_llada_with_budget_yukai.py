class LLaDALlamaBlock(LLaDABlock):

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
                q_denoising, k, v,
                attention_bias=attention_bias,
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

'''[current] = [refresh|denoising]'''
def attention(
    self,
    q_denoising: torch.Tensor,
    k_current: torch.Tensor,
    v_current: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    attention_bias: Optional[torch.Tensor] = None,
    layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    use_cache: bool = False,
    idx_refresh: Optional[torch.Tensor] = None,
    idx_denoising: Optional[torch.Tensor] = None,
    shape_target: Optional[Tuple[int, int, int]] = None

) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
    
    B, T_q, C = q_denoising.size()  # batch size, sequence length, d_model
    T_kv = k_current.shape[1]
    dtype = k_current.dtype

    # Optionally apply layer norm to keys and queries.
    if self.q_norm is not None and self.k_norm is not None: #self.q_norm: None, self.k_norm: None
        q_denoising = self.q_norm(q_denoising).to(dtype=dtype)
        k_current = self.k_norm(k_current).to(dtype=dtype)
    # end

    # Move head forward to be next to the batch dim.
    # shape: (B, nh, T, hs)
    # self.config.n_heads: 32
    q_denoising = q_denoising.view(B, T_q, self.config.n_heads, C // self.config.n_heads).transpose(1, 2)
    # shape: (B, n_kv_h, T, hs)
    k_current = k_current.view(B, T_kv, self.config.effective_n_kv_heads, C // self.config.n_heads).transpose(1, 2)
    # shape: (B, n_kv_h, T, hs)
    v_current = v_current.view(B, T_kv, self.config.effective_n_kv_heads, C // self.config.n_heads).transpose(1, 2)

    if not layer_past:
        raise NotImplementedError('no layer_past case is current not supported.')
    # end

    k_previous, v_previous = layer_past

    k_final = concat_and_replace(k_previous, k_current, idx_refresh, idx_denoising, shape_target)
    v_final = concat_and_replace(v_previous, v_current, idx_refresh, idx_denoising, shape_target)

    max_replace_pos = k_final.shape[1]
    q_denoising_rotated, k_final_rotated = self.rotary_emb(q_denoising, k_final, max_replace_pos)   # WARNING: might have problem

    hidden = self._scaled_dot_product_attention(
        q_denoising_rotated, 
        k_final_rotated,
        v_final,
        attn_mask=None,
        dropout_p=0.0 if not self.training else self.config.attention_dropout,
        is_causal=False
    )

    hidden = hidden.transpose(1,2).contiguous().view(B,T_q,C)
    return self.attn_out(hidden), (k_final, v_final)
# end