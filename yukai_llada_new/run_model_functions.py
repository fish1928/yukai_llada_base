import torch

import json
from tqdm import tqdm
from tools_llada import PPLCalculator, RefreshIdxHelper

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

    for id_block in range(num_blocks):
        position_end = position_start + len_prompt + (id_block+1) * size_block
        mask_mask_blk = x[:,position_start:position_end] == id_mask
        
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

    '''accumulate only'''
    id_batch = kwargs['id_batch']
    calculator_kvsim = kwargs['calculator_kvsim']
    '''accumulate only'''
    
    p_finalized = torch.zeros(x.shape, dtype=torch.float64, device=x.device)
    position_start = 0

    for id_block in range(num_blocks):
        position_end = position_start + len_prompt + (id_block+1) * size_block
        mask_mask_blk = x[:,position_start:position_end] == id_mask

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

            '''accumulate only'''
            calculator_kvsim.collect_kv_previous_and_calculate_sim_per_step_()  # DIFF: this line is the only difference
            '''accumulate only'''
        # end for step
    # end for block

    return p_finalized[:, len_prompt:]
# end


@ torch.no_grad()
def run_model_semi_cached(model, x, y, config_diffusion, *args, **kwargs):

    '''declare required variables'''
    num_blocks = config_diffusion.num_blocks
    step_per_block = config_diffusion.step_per_block
    size_block = config_diffusion.size_block
    id_mask = config_diffusion.id_mask
    len_prompt = config_diffusion.len_prompt
    sorter = config_diffusion.klass_sorter()
    collector = config_diffusion.klass_collector()
    
    p_finalized = torch.zeros(x.shape, dtype=torch.float64, device=x.device)

    idx_denoising = torch.arange(0, len_prompt, dtype=torch.long).to(x.device)
    model(x[:, idx_denoising], idx_current=idx_denoising)   # save prompt for once, shape_target can be overlook the first time

    for id_block in range(num_blocks):
        position_start = len_prompt + id_block * size_block
        position_end = position_start + size_block
        mask_mask_blk = x[:,position_start:position_end] == id_mask

        idx_denoising = torch.arange(position_start, position_end, dtype=torch.long).to(x.device)
        quota_helper = BlockDiffusionQuotaHelper(mask_mask_blk, size_block)

        for step in range(step_per_block):
            x_denoising,  y_denoising= x[:, idx_denoising], y[:, idx_denoising]
            shape_target = (x.shape[0], position_end, -1)
            logits = model(x_denoising, idx_current=idx_denoising, shape_target=shape_target).logits
            snapshot = SimpleLogitsSnapshot(logits, x_denoising, y_denoising, id_mask)
            
            conf_snapshot = snapshot.transform_logits(collector)
            idx_sorted_by_conf = sorter.argsort(conf_snapshot, snapshot)
            num_unmask = quota_helper.get_quota(step)
            idx_transform = idx_sorted_by_conf[:, :num_unmask]

            snapshot.materialize_by_idx_(idx_transform, conf_snapshot)
            snapshot.update_this(1, idx_src=idx_transform, idx_tgt=idx_denoising, y=x).update_this(1, idx_src=idx_transform, idx_tgt=idx_denoising, p_finalized=p_finalized)
        # end for step
    # end for block

    return p_finalized[:, len_prompt:]
# end


@ torch.no_grad()
def run_model_semi_cached_snapshot_refresh_one(model, x, y, config_diffusion, *args, **kwargs):

    '''declare required variables'''
    num_blocks = config_diffusion.num_blocks
    step_per_block = config_diffusion.step_per_block
    size_block = config_diffusion.size_block
    id_mask = config_diffusion.id_mask
    len_prompt = config_diffusion.len_prompt
    sorter = config_diffusion.klass_sorter()
    collector = config_diffusion.klass_collector()

    idx_refresh = torch.tensor([], dtype=torch.long, device=x.device)
    p_finalized = torch.zeros(x.shape, dtype=torch.float64, device=x.device)

    position_start, position_end = 0, len_prompt
    idx_denoising = torch.arange(position_start, position_end, dtype=torch.long, device=x.device)
    idx_current = torch.cat([idx_refresh, idx_denoising])
    shape_target = (x.shape[0], position_end, -1)
    logits = model(x[:, idx_current], idx_current=idx_current, shape_target=shape_target).logits
    snapshot = SimpleLogitsSnapshot(logits, x[:, idx_current], y[:, idx_current], id_mask)

    for id_block in range(num_blocks):
        position_start = len_prompt + id_block * size_block
        position_end = position_start + size_block
        mask_mask_blk = x[:,position_start:position_end] == id_mask
        quota_helper = BlockDiffusionQuotaHelper(mask_mask_blk, size_block)

        idx_denoising = torch.arange(position_start, position_end, dtype=torch.long).to(x.device)
        idx_current = torch.cat([idx_refresh, idx_denoising]) 
        shape_target = (x.shape[0], position_end, -1)
        logits = model(x[:, idx_current], idx_current=idx_current, shape_target=shape_target).logits

        logits_denoising = logits[:, -idx_denoising.shape[-1]:]
        logits_accumulated = torch.cat([snapshot.get_logits(), logits_denoising], dim=1)
        x_accumulated = x[:, :position_end]
        y_accumulated = y[:, :position_end]

        # update snapshot
        snapshot = SimpleLogitsSnapshot(logits_accumulated, x_accumulated, y_accumulated, id_mask)

        for step in range(step_per_block):
            conf_snapshot = snapshot.transform_logits(collector)    # 全的
            idx_sorted_by_conf = sorter.argsort(conf_snapshot, snapshot)    # 全的
            num_unmask = quota_helper.get_quota(step)
            idx_transform_2d = idx_sorted_by_conf[:, :num_unmask]     # 全的(2d)

            idx_current = torch.cat([idx_refresh, idx_transform_2d.squeeze(0)], dim=-1)
            logits = model(x[:, idx_current], idx_current=idx_current, shape_target=shape_target).logits
            logits_transform = logits[:, -idx_transform_2d.shape[-1]:]

            snapshot.update_logits_(idx_transform_2d, logits_transform)
            conf_snapshot = snapshot.transform_logits(collector)
            snapshot.materialize_by_idx_(idx_transform_2d, conf_snapshot)

            idx_refresh = idx_transform_2d.squeeze(0)
            snapshot.update_this(1, idx_src=idx_transform_2d, y=x).update_this(1, idx_src=idx_transform_2d, p_finalized=p_finalized)
        # end for step
    # end for block

    return p_finalized[:, len_prompt:]
# end





class RunModelCacheWithRefresh:

    @ torch.no_grad()
    def run_model_semi_cached_refresh(self, model, x, y, config_diffusion, *args, **kwargs):

        '''declare required variables'''
        num_blocks = config_diffusion.num_blocks
        step_per_block = config_diffusion.step_per_block
        size_block = config_diffusion.size_block
        id_mask = config_diffusion.id_mask
        len_prompt = config_diffusion.len_prompt
        sorter = config_diffusion.klass_sorter()
        collector = config_diffusion.klass_collector()
        refresher = kwargs['refresher']

        p_finalized = torch.zeros(x.shape, dtype=torch.float64, device=x.device)
        idx_denoising = torch.arange(0, len_prompt, dtype=torch.long).to(x.device)
        model(x[:, idx_denoising], idx_current=idx_denoising)   # save prompt for once

        for id_block in range(num_blocks):
            position_start = len_prompt + id_block * size_block
            position_end = position_start + size_block
            mask_mask_blk = x[:,position_start:position_end] == id_mask
            idx_denoising = torch.arange(position_start, position_end, dtype=torch.long).to(x.device)
            quota_helper = BlockDiffusionQuotaHelper(mask_mask_blk, size_block)

            for step in range(step_per_block):

                idx_refresh = refresher.get_refresh_idx(x, step, id_block, return_sorted=True)
                idx_current = torch.cat([idx_refresh, idx_denoising])
                shape_target = (x.shape[0], position_end, -1)
                x_current, x_denoising,  y_denoising= x[:, idx_current], x[:, idx_denoising], y[:, idx_denoising]

                logits_current = model(x_current, idx_current=idx_current, shape_target=shape_target).logits
                logits_denoising = logits_current[:, -idx_denoising.shape[-1]:]
                snapshot = SimpleLogitsSnapshot(logits_denoising, x_denoising, y_denoising, id_mask)
                
                conf_snapshot = snapshot.transform_logits(collector)
                idx_sorted_by_conf = sorter.argsort(conf_snapshot, snapshot)
                num_unmask = quota_helper.get_quota(step)
                idx_transform = idx_sorted_by_conf[:, :num_unmask]

                snapshot.materialize_by_idx_(idx_transform, conf_snapshot)
                snapshot.update_this(1, idx_src=idx_transform, idx_tgt=idx_denoising, y=x).update_this(1, idx_src=idx_transform, idx_tgt=idx_denoising, p_finalized=p_finalized)
            # end for step
        # end for block

        return p_finalized[:, len_prompt:]
    # end

    def main(self, model, config):
        filename = 'all_in_one_sim_report_abs_per_step_p.json'
        with open(filename, 'r') as file:
            data_refresh = json.load(file)
        # end

        refresher = RefreshIdxHelper(
            data_refresh,
            'v',
            config.size_block,
            randomed=False
        )

        calculator_ppl = PPLCalculator()
        model.fill_plugin(config.klass_cache_past_kv).fill_plugin(config.klass_save_kv_previous)
        plugin_cache_past_kv = config.klass_cache_past_kv()

        '''start the evaluation process'''
        for id_batch, batch in enumerate(tqdm(loader)):
            plugin_cache_past_kv.clear(model)
            refresher.set_budget(1).set_sample_id(id_batch)

            conf = run_model_semi_cached_refresh(
                model,
                batch['ids_prompt_masked_full'].to(config.device),
                batch['ids_target_masked_full'].to(config.device),
                config,
                refresher=refresher
            )

            print(calculator_ppl.cal(conf))
        # end for        
    # end
# end