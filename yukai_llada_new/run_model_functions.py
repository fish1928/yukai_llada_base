@ torch.no_grad()
def run_model_semi(model, x, y, config_diffusion, *args, **kwargs):

    '''declare required variables'''
    num_blocks = config_diffusion.num_blocks
    step_per_block = config_diffusion.step_per_block
    size_block = config_diffusion.size_block
    id_mask = config_diffusion.id_mask
    len_prompt = config_diffusion.len_prompt
    sorter = config_diffusion.klass_sorter()
    collector = config_diffusion.klass_collector()
    
    p_finalized = torch.zeros(x.shape, dtype=torch.float64, device=x.device)
    position_start = 0

    for id_block in range(1, num_blocks+1):
        position_end = position_start + len_prompt + id_block * size_block
        mask_mask_blk = x[:,position_start:position_end] == id_mask

        B = x.shape[0]
        L = position_end - position_start
        
        idx_denoising = torch.arange(position_start, position_end, dtype=torch.long).to(x.device)
        quota_helper = BlockDiffusionQuotaHelper(mask_mask_blk, size_block)

        for step in range(step_per_block):
            x_denoising,  y_denoising= x[:, idx_denoising], y[:, idx_denoising]
            logits = model(x_denoising, idx_current=idx_denoising).logits
            snapshot = SimpleLogitsSnapshot(logits, x_denoising, y_denoising, id_mask)
            conf_snapshot = snapshot.transform_logits(collector)
            idx_sorted_by_conf = sorter.argsort(conf_snapshot, snapshot)
            num_unmask = quota_helper.get_quota(step)
            idx_transform = idx_sorted_by_conf[:, :num_unmask]

            snapshot.materialize_by_idx_(idx_transform, conf_snapshot)
            snapshot.update_this(1, idx_transform, y=x).update_this(1, idx_transform, p_finalized=p_finalized)
            
        # end for step
    # end for block

    return p_finalized[:, len_prompt:]
# end


@ torch.no_grad()
def run_model_semi_collect_kv(model, x, y, config_diffusion, *args, **kwargs):

    '''declare required variables'''
    num_blocks = config_diffusion.num_blocks
    step_per_block = config_diffusion.step_per_block
    size_block = config_diffusion.size_block
    id_mask = config_diffusion.id_mask
    len_prompt = config_diffusion.len_prompt
    sorter = config_diffusion.klass_sorter()
    collector = config_diffusion.klass_collector()

    id_batch = kwargs['id_batch']
    calculator_kvsim = kwargs['calculator_kvsim']
    
    p_finalized = torch.zeros(x.shape, dtype=torch.float64, device=x.device)
    position_start = 0

    for id_block in range(1, num_blocks+1):
        position_end = position_start + len_prompt + id_block * size_block
        mask_mask_blk = x[:,position_start:position_end] == id_mask

        B = x.shape[0]
        L = position_end - position_start
        
        idx_denoising = torch.arange(position_start, position_end, dtype=torch.long).to(x.device)
        quota_helper = BlockDiffusionQuotaHelper(mask_mask_blk, size_block)

        for step in range(step_per_block):
            x_denoising,  y_denoising= x[:, idx_denoising], y[:, idx_denoising]
            logits = model(x_denoising, idx_current=idx_denoising).logits
            snapshot = SimpleLogitsSnapshot(logits, x_denoising, y_denoising, id_mask)
            conf_snapshot = snapshot.transform_logits(collector)
            idx_sorted_by_conf = sorter.argsort(conf_snapshot, snapshot)
            num_unmask = quota_helper.get_quota(step)
            idx_transform = idx_sorted_by_conf[:, :num_unmask]

            snapshot.materialize_by_idx_(idx_transform, conf_snapshot)
            snapshot.update_this(1, idx_transform, y=x).update_this(1, idx_transform, p_finalized=p_finalized)
            calculator_kvsim.collect_kv_previous_and_calculate_sim_per_step_()  # DIFF: this line is the only difference
        # end for step
    # end for block

    return p_finalized[:, len_prompt:]
# end