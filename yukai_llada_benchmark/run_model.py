import torch
from tqdm import tqdm

from components_llada import SimpleLogitsSnapshot
from tools_llada import BlockDiffusionQuotaHelper
from plugins_llada import SaveKVPreviousPlugin_Disabled, SaveKVPreviousPlugin_Enabled,\
                            CachePastKVPlugin_Disabled, CachePastKVPlugin_Enabled,\
                            CacheAttnPlugin_Disabled, CacheAttnPlugin_Enabled,\
                            CacheVOPlugin_Disabled, CacheVOPlugin_Enabled


# model runner
class RunModelSemiCached:

    def config_plugin_(self, config):
        config.klass_save_kv_previous=SaveKVPreviousPlugin_Disabled,
        config.klass_cache_past_kv=CachePastKVPlugin_Enabled,
        config.klass_cache_attn=CacheAttnPlugin_Enabled,
        config.klass_cache_vo=CacheVOPlugin_Disabled

        return self
    # end

    def register_plugin_(self, model, config):
        model\
            .fill_plugin(config.klass_cache_past_kv)\
            .fill_plugin(config.klass_save_kv_previous)\
            .fill_plugin(config.klass_cache_attn)\
            .fill_plugin(config.klass_cache_vo)
        # end
    # end

    #TODO: 有一个y的问题
    @ torch.no_grad()
    def generate(self, model, tokenizer, config_diffusion, *args, **kwargs):

        '''declare required variables'''
        num_blocks = config_diffusion.num_blocks
        step_per_block = config_diffusion.step_per_block
        size_block = config_diffusion.size_block
        id_mask = config_diffusion.id_mask
        sorter = config_diffusion.klass_sorter()
        collector = config_diffusion.klass_collector()

        words_stop = kwargs['until']
        text_prompt = kwargs['text_prompt']
        len_prompt = kwargs['len_prompt']

        x = kwargs['ids_input']

        has_done = False
        sentence_all = text_prompt

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
                snapshot.update_this(1, idx_transform, x=x)
            # end for step

            sentence_block_current = tokenizer.batch_decode(x[:, idx_denoising])[0]

            for word_stop in words_stop:
                if word_stop in sentence_block:
                    sentence_block = sentence_block_current.lsplit(word_stop)[0]
                    has_done = True
                # end
            # end

            if has_done:
                sentence_block_previous = tokenizer.batch_decode(x[:, len_prompt:position_start])[0]
                sentence_all += sentence_block_previous + sentence_block_current
                return sentence_all, has_done
            # end if
        # end for

        sentence_remain = tokenizer.batch_decode(x[:, len_prompt:])[0]
        sentence_all += sentence_remain

        return sentence_all, has_done
    # end

    def run_one(self, model, tokenizer, config, *args, **kwargs):

        plugin_cache_past_kv = self.config.klass_cache_past_kv()
        plugin_cache_past_kv.clear(self.model)

        sentence_generated, has_done = self.generate(
            model,
            tokenizer,
            config,
            *args,
            **kwargs
        )

        return sentence_generated, has_done
    # end
# end

class RunModelSemiCachedWithMLP:
    def generate(self, model, x, config_diffusion, *args, **kwargs):
        '''declare required variables'''
        num_blocks = config_diffusion.num_blocks
        step_per_block = config_diffusion.step_per_block
        size_block = config_diffusion.size_block
        id_mask = config_diffusion.id_mask
        len_prompt = config_diffusion.len_prompt
        sorter = config_diffusion.klass_sorter()
        collector = config_diffusion.klass_collector()
        
        words_stop = config_diffusion.words_stop
        tokenizer = config_diffusion.tokenizer
        text_prompt = config_diffusion.text_prompt

        plugin_cache_attn = kwargs['plugin_cache_attn']
        step_refresh_remainder = kwargs['step_refresh_remainder']
        future_idx_selector = kwargs['future_idx_selector'] # budget is also here

        has_done = False
        sentence_all = text_prompt

        idx_refresh = torch.tensor([], dtype=torch.long, device=x.device)

        position_start, position_end = 0, len_prompt
        idx_denoising = torch.arange(position_start, position_end, dtype=torch.long, device=x.device)
        idx_current = torch.cat([idx_refresh, idx_denoising])
        shape_target = (x.shape[0], position_end, -1)
        logits = model(x[:, idx_current], idx_current=idx_current, shape_target=shape_target).logits
        snapshot = SimpleLogitsSnapshot(logits, x[:, idx_current], y[:, idx_current], id_mask)

        for id_block in range(num_blocks):
            position_start = len_prompt + id_block * size_block
            position_end = position_start + size_block
            mask_mask_block = x[:,position_start:position_end] == id_mask
            quota_helper = BlockDiffusionQuotaHelper(mask_mask_block, size_block)

            idx_block = torch.arange(position_start, position_end, dtype=torch.long, device=x.device)
            shape_target = (x.shape[0], position_end, -1)

            for step in range(step_per_block):

                if step == 0 or step % step_refresh_remainder == 0:
                    idx_denoising = idx_block

                    if step == 0:
                        idx_current = torch.cat([idx_refresh, idx_denoising])   # only the first time need refresh previous
                    else:
                        idx_current = idx_denoising
                    # end

                    logits = model(x[:, idx_current], idx_current=idx_current, shape_target=shape_target).logits
                    logits_denoising = logits[:, -idx_denoising.shape[-1]:]

                    logits_accumulated = torch.cat([snapshot.get_logits()[:, :position_start, :], logits_denoising], dim=1)
                    x_accumulated = x[:, :position_end]
                    snapshot = SimpleLogitsSnapshot(logits_accumulated, x_accumulated, x_accumulated, id_mask)
                    conf_snapshot = snapshot.transform_logits(collector)
                else:
                    score_attn = plugin_cache_attn.collect_attn_from_all_blocks(model)
                    idx_in_attn = torch.where(idx_transform_2d.squeeze(0) == idx_current)[0]    # idx_current is now last round
                    score_attn = score_attn[-1, idx_in_attn, -idx_block.shape[-1]:].squeeze(1)
                    mask_mask_current_no = ~(x[:,position_start:position_end] == id_mask).view(1,-1)    # (B, K)
                    score_attn.masked_fill_(mask_mask_current_no, torch.finfo(score_attn.dtype).min)

                    idx_denoising = (future_idx_selector.select_future_by_attn(score_attn) + position_start).squeeze(0)
                    idx_current = torch.cat([idx_refresh, idx_denoising])

                    logits = model(x[:, idx_current], idx_current=idx_current, shape_target=shape_target).logits
                    logits_transform = logits[:, -idx_denoising.shape[-1]:]

                    # different here compared to step == 0
                    snapshot.update_logits_(idx_denoising.unsqueeze(0), logits_transform)
                    conf_snapshot = snapshot.transform_logits(collector)
                    # different ends

                    if future_idx_selector.select_only_in_h: #TODO: be careful of the use of scatter(shape)
                        mask_denoising_no = ~torch.isin(torch.arange(conf_snapshot.shape[-1], device=conf_snapshot.device), idx_denoising).unsqueeze(0)    # idx_denoising -> 
                        conf_snapshot.masked_fill_(mask_denoising_no, torch.finfo(conf_snapshot.dtype).min)
                    # end

                # end

                idx_sorted_by_conf = sorter.argsort(conf_snapshot, snapshot)    # truth
                num_unmask = quota_helper.get_quota(step)
                idx_transform_2d = idx_sorted_by_conf[:, :num_unmask]
                snapshot.materialize_by_idx_(idx_transform_2d, conf_snapshot)
                snapshot.update_this(1, idx_transform_2d, x=x)
                idx_refresh = idx_transform_2d.squeeze(0)
            # end

            sentence_block_current = tokenizer.batch_decode(x[:, idx_denoising])[0]

            for word_stop in words_stop:
                if word_stop in sentence_block:
                    sentence_block = sentence_block_current.lsplit(word_stop)[0]
                    has_done = True
                # end
            # end

            if has_done:
                sentence_block_previous = tokenizer.batch_decode(x[:, len_prompt:position_start])[0]
                sentence_all += sentence_block_previous + sentence_block_current
                return sentence_all, has_done
            # end if
        # end for
        sentence_remain = tokenizer.batch_decode(x[:, len_prompt:])[0]
        sentence_all += sentence_remain

        return sentence_all, has_done
    # end function
# end class