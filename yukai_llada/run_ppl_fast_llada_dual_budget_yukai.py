import torch

@torch.no_grad()
def run_model_with_dual_cache(
    model,
    ids_input_masked_full,
    ids_target_masked_full,
    len_prompt,
    remasking='truth_top_k',
    steps=128,
    gen_length=128,
    block_length=128,
    temperature=0.,
    mask_id=126336,
    is_eval=True
):
    
    x, y = ids_input_masked_full, ids_target_masked_full
    B = x.shape[0]

    probs_all = torch.zeros(x.shape, dtype=torch.float64).to(model.device)

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps_per_block = steps // num_blocks

    x_prompt = x[:, :len_prompt]
    idx_prompt = torch.arange(x_prompt.shape[1])
    out_prompt = model(x_prompt, use_cache=True, idx_denoising=idx_prompt)
    past_key_values = out_prompt.past_key_values

    count_step_diffusion = 0

    for id_block in range(num_blocks):

        position_start = len_prompt + id_block * block_length
        position_end = position_start + block_length
        block_mask_index = (x[:, position_start:position_end] == mask_id)  # (B, block_length)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps_per_block)  # (B, steps_per_block)

        # Build a replace_position tensor indicating the block range (static slice)
        mask_position_replace = torch.zeros_like(x, dtype=torch.bool)
        mask_position_replace[:, position_start:position_end] = True  # boolean mask (not a dynamic slice bound)

        for step_in_block in range(steps_per_block):
            # Evaluate logits only for current block with cache
            if (x[:, position_start:position_end] == mask_id).sum() == 0:
                break
            # end

            idx_refresh = get_refresh_idx()
            idx_denoising = torch.arange(position_start, position_end,dtype=torch.long)
            idx_current = torch.cat([idx_refresh, idx_denoising])
            x_current = x[:, idx_current]

            output_blk = model(
                x_current,
                past_key_values=past_key_values,
                use_cache=True,
                idx_refresh=idx_refresh,
                idx_denoising=idx_denoising,
                shape_target=(x.shape[0], position_end, -1)
            )

            logits_blk = output_blk.logits
            past_key_values=output_blk.past_key_values  # update past_key_values here

            # Mask and quota for this step (all tensor ops)
            mask_blk = (x[:, position_start:position_end] == mask_id)  # (B, block_length)
            blk_x = x[:, position_start:position_end]
            blk_y = y[:, position_start:position_end]
            blk_prob = probs_all[:, position_start:position_end]

            quota_i = num_transfer_tokens[:, step_in_block]  # (B,)
            blk_x0, blk_x0_p, transfer_idx_blk = get_transfer_index(
                logits_blk,
                temperature,
                remasking,
                mask_blk,
                blk_x,
                blk_y,
                quota_i
            )

            if is_eval:
                blk_x[transfer_idx_blk] = blk_y[transfer_idx_blk]
                blk_prob[transfer_idx_blk] = blk_x0_p[transfer_idx_blk]
            else:
                blk_x[transfer_idx_blk] = blk_x0[transfer_idx_blk]
            # end

            count_step_diffusion += 1

    return probs_all, y != mask_id
# end