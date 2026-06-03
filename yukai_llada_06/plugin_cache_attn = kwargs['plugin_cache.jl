plugin_cache_attn = kwargs['plugin_cache_attn']


def get_attn(plugin_cache_attn):    
    num_unmask_total = sum([quota_helper.get_quota(step+i) for i in range(min(step_per_block - step, step_refresh_remainder))])
    idx_transform_total_2d = idx_sorted_by_conf[:, :num_unmask_total]     # 全的(2d)
    idx_transform_total_1d = idx_transform_total_2d.squeeze(0)

    scores_attn_all = plugin_cache_attn.collect_attn_from_all_blocks(model) # (B, Blk, K)
    idx_query = idx_transform_total_1d - position_start
    scores_attn = scores_attn_all[:, idx_query, :] #(B, Q, K))

    del scores_attn_all
    scores_key = scores_attn.mean(dim=(0, 1))          # (seq_k,) -> 3个? 有问题
    scores_key[idx_transform_total_1d] = 0.0
    scores_key = torch.where(torch.arange(idx_sorted_by_conf.shape[-1], device=x.device)>len_prompt, scores_key, 0.0)
    idx_most_attended_keys = torch.argsort(scores_key, descending=True)[:budget_refresh]

    idx_transform_previous_1d = idx_transform_2d.squeeze(0)
    # torch.Size([3]) torch.Size([1, 1]) torch.Size([1])
    # jprint(idx_most_attended_keys.shape, idx_transform_2d.shape, idx_transform_previous_1d.shape)
    idx_refresh = torch.cat([idx_most_attended_keys, idx_transform_previous_1d[-idx_transform_previous_1d.shape[0]:]])
    del idx_transform_previous_1d
# end

plugin_cache_attn = config.klass_cache_attn()
plugin_cache_attn.clear(model)
plugin_cache_attn=plugin_cache_attn,


stats.margin.add(step_global, margin_p)
stats.conf.add(step_global, conf_snapshot)
stats.attn.add(step_global, attn)
stats.unmask.add(step_global, idx_transform.squeeze(0))


The size of tensor a (96) must match the size of tensor b (126464) at non-singleton dimension 1


torch.Size([1, 96, 126464]) torch.Size([1, 96, 126464]) torch.Size([1, 126464])
torch.Size([1, 126464]) torch.Size([1, 96])