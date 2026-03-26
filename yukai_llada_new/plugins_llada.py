import os
import inspect
from abc import ABC, abstractmethod

from collections import defaultdict

import torch
import torch.nn.functional as F
import json


from tools_debug import jprint

class InspectorPlugin(ABC):

    @abstractmethod
    def get_plugin_name(self):
        raise NotImplementedError
    # end

    def load_vars(self, *args):
        frame = inspect.currentframe().f_back.f_back
        locals_caller = frame.f_locals
        return tuple(locals_caller[arg] for arg in args)
    # end

    def load_attrs(self, *args):
        frame = inspect.currentframe().f_back.f_back
        vars_caller = frame.f_locals
        self_caller = vars_caller['self']
        return tuple(getattr(self_caller, arg, None) for arg in args )
    # end

    def load_func(self, arg):
        frame = inspect.currentframe().f_back.f_back
        vars_caller = frame.f_locals
        self_caller = vars_caller['self']
        return getattr(self_caller, arg)
    # end

    def save_attrs(self, **kvargs):
        frame = inspect.currentframe().f_back.f_back
        vars_caller = frame.f_locals
        self_caller = vars_caller['self']
        for k, v in kvargs.items():
            setattr(self_caller, k, v)
        # end     
    # end

    def __bool__(self):
        name_klass = self.__class__ # <class '__main__.A_Enabled'>
        str_enabled = str(name_klass).split('.')[-1].split("'")[0].split('_')[-1].lower()

        return str_enabled == 'enabled'
    # end
# end

class CachePastKVPlugin_Disabled(InspectorPlugin):

    def get_plugin_name(self):
        return 'cache_past_kv'
    # end

    def load(self):
        k_final, v_final = self.load_vars('k_current', 'v_current')
        return k_final, v_final
    # end

    def save(self):
        pass
    # end
# end

class CachePastKVPlugin_Enabled(InspectorPlugin):

    def get_plugin_name(self):
        return 'cache_past_kv'
    # end

    def load(self):
        layer_past = self.load_attrs('layer_past')[0]

        k_current, v_current = self.load_vars('k_current', 'v_current')
        
        if layer_past is None:  # the first time
            k_final, v_final = k_current, v_current
            return k_final, v_final
        # end

        concat_and_replace = self.load_func('concat_and_replace')
        idx_current, shape_target = self.load_vars('idx_current','shape_target')

        k_previous, v_previous = layer_past

        k_final = concat_and_replace(k_previous, k_current, idx_current, shape_target)
        v_final = concat_and_replace(v_previous, v_current, idx_current, shape_target)
        
        return k_final, v_final
    # end

    def save(self):
        k_final, v_final = self.load_vars('k_final', 'v_final')
        layer_past = (k_final, v_final)
        self.save_attrs(layer_past=layer_past)
    # end

    def clear(self, model):
        for block in model.model.transformer.blocks:
            if hasattr(block, 'layer_past'):
                del block.layer_past
            # end
        # end
    # end
# end


class SaveKVPreviousPlugin_Disabled(InspectorPlugin):

    def get_plugin_name(self):
        return 'save_kv_previous'
    # end

    def refresh(self):
        pass
    # end

    def save(self):
        pass
    # end

# end

class SaveKVPreviousPlugin_Enabled(InspectorPlugin):
    
    def get_plugin_name(self):
        return 'save_kv_previous'
    # end

    def refresh(self):
       self.save_attrs(_k_previous=None, _v_previous=None)
    # end

    def save(self):
        k, v = self.load_vars('k','v')
        self.save_attrs(_k_previous=k, _v_previous=v)
    # end

    def clear(self, model):
        for block in model.model.transformer.blocks:
            if hasattr(block, '_k_previous'):
                del block._k_previous
            # end

            if hasattr(block, '_v_previous'):
                del block._v_previous
            # end            
        # end
    # end

    '''aggregation and calculation'''

    def _get_names_hidden(self):
        return ['_k_previous','_v_previous']
    # end


    def __init__(self):
        self.dict_hidden_to_matrixs_sim_per_step = {}
        for name_hidden in self._get_names_hidden():
            self.dict_hidden_to_matrixs_sim_per_step[name_hidden] = []
        # end

        self.dict_cache_kv_previous = {}
    # end

    def collect_kv_previous_and_calculate_sim_per_step_(self):
        id_batch, model, x = self.load_vars('id_batch', 'model', 'x')

        dict_hidden_to_sims_layer = {}
        for name_hidden in self._get_names_hidden():
            dict_hidden_to_sims_layer[name_hidden] = []
        # end

        for block_transformer in model.model.transformer.blocks[:]:                       # take last all layers
            id_block_transformer = block_transformer.layer_id
            name_cache_base = f'batch_{id_batch}_layer_{id_block_transformer}'  # block and step in block

            for name_hidden in self._get_names_hidden():
                if hasattr(block_transformer, name_hidden):
                    cache_current = getattr(block_transformer, name_hidden)
                    name_cache = f'{name_cache_base}.{name_hidden}'

                    if name_cache not in self.dict_cache_kv_previous:
                        self.dict_cache_kv_previous[name_cache] = cache_current
                        continue
                    # end

                    # we have current and last, calculate similarity
                    cache_last = self.dict_cache_kv_previous[name_cache]
                    self.dict_cache_kv_previous[name_cache] = cache_current  # udpate cache

                    if cache_last.shape[1] < cache_current.shape[1]:
                        cache_last = torch.cat([cache_last, cache_current[:, cache_last.shape[1]:, :]], dim=1)
                    # end

                    sim_neighbour = F.cosine_similarity(cache_current, cache_last, dim=-1).clamp(-1.0, 1.0)
                    
                    if sim_neighbour.shape[-1] < x.shape[-1]:
                        sim_neighbour_padded = F.pad(
                            sim_neighbour,
                            (0, x.shape[-1]-sim_neighbour.shape[1]),
                            value=1.0
                        ).squeeze(0)
                    else:
                        sim_neighbour_padded = sim_neighbour.squeeze(0)
                    # end

                    dict_hidden_to_sims_layer[name_hidden].append(sim_neighbour_padded)
                # end
            # end
        # end

        for name_hidden in self._get_names_hidden():
            sims_layer = dict_hidden_to_sims_layer[name_hidden]

            if len(sims_layer) == 0:
                break
            # end
            
            matrix_sim_per_step = torch.stack(sims_layer, dim=0)
            self.dict_hidden_to_matrixs_sim_per_step[name_hidden].append(matrix_sim_per_step)
        # end

        return self
    # end

    def aggregate_result_(self):
        self.result = {}

        for name_hidden in self._get_names_hidden():
            matrixs_sim_per_step = self.dict_hidden_to_matrixs_sim_per_step[name_hidden]
            matrix_sim_per_step_layer_token = torch.stack(matrixs_sim_per_step, 0)  # dimension
            self.result[name_hidden] = matrix_sim_per_step_layer_token.detach().float().cpu()
        # end for

        return self
    # end

    def dump_result_to_file(self, id_batch, folder_output):
        result = self.result
        os.makedirs(folder_output, exist_ok=True)

        for name_hidden, matrix_sim_per_step_layer_token in result.items():
            filename_sim_final = f'batch_{id_batch}{name_hidden}.pt'
            path_file_sim_final = os.path.join(folder_output, filename_sim_final)
            print(f'saving {path_file_sim_final} with shape {matrix_sim_per_step_layer_token.shape}')
            torch.save(matrix_sim_per_step_layer_token, path_file_sim_final)
        # end
    # end

    '''further calculation'''

    def token_nonsimilarity_score_abs_per(
        self,
        sim: torch.Tensor,
        p: float = 3.0,
        type_fn: str = 'p',
        type_aggregate: str = 'step'
    ) -> torch.Tensor:

        assert sim.ndim == 3, f"Expected 3D tensor [steps, layers, tokens], got shape {tuple(sim.shape)}"
        S, L, T = sim.shape

        dim_aggregate = 1 if type_aggregate == 'step' else (0, 1)

        diff = torch.abs(1.0 - sim)
        if type_fn == 'p':
            score = diff.pow(p).mean(dim=dim_aggregate).pow(1.0 / p)
        elif type_fn == 'log':
            score = torch.log1p(diff).mean(dim=dim_aggregate)
        # end

        if score.dim() == 1:
            score = score.view(1, -1).expand(S, -1)
        # end

        return score
    # end

    def load_sim_matrix_and_transform_to_most_diff_list_per(self, folder_kv_base, filename, num_block, len_prompt, size_block, type_aggregate='step'):
        path_kv_file = os.path.join(folder_kv_base, filename)
        matrix_sim_step_layer_token = torch.load(path_kv_file)
        matrix_sim_step_layer_token = F.pad(matrix_sim_step_layer_token, (0,0,0,0,1,0), value=1.0)

        list_idx_diff_sorted = []

        for id_block in range(num_block):
            pos_end_dim_t = len_prompt + size_block * id_block # cache end
            pos_start_dim_s = id_block * size_block

            matrix_sim_step_layer_token_cached = matrix_sim_step_layer_token[pos_start_dim_s:pos_start_dim_s+size_block, :, :pos_end_dim_t]  #(steps_block, len_cached)
            matrix_step_scores_diff_token = self.token_nonsimilarity_score_abs_per(matrix_sim_step_layer_token_cached, type_aggregate=type_aggregate)    # (1, len_cached)

            matrix_step_idx_diff_token_decending = torch.argsort(matrix_step_scores_diff_token, dim=-1, descending=True)    # (1, len_cached)

            for step in range(matrix_step_idx_diff_token_decending.shape[0]):
                idxs_diff_token_decending = matrix_step_idx_diff_token_decending[step,:]    # (len_cached)

                list_idx_diff_sorted.append({'filename': filename, 'block': id_block, 'step': step, 'idx': idxs_diff_token_decending.tolist(), 'value_raw': matrix_step_scores_diff_token[step,:].tolist()})
            # end
        # end

        return list_idx_diff_sorted
    # end


    '''
        folder_kv_base = 'sims_kv_0315'
        type_fn = 'p'
        type_aggregate = 'block'
        stamp = '0326'
        len_prompt = 512
        num_block = 8
        len_target = 1024
    '''
    def dump_all_in_one(
            self,
            folder_kv_base,
            len_prompt,
            len_target,
            num_blocks,
            type_fn,
            type_aggregate,
            stamp
    ):  # from test_get_top_change.ipynb
        size_block = int(len_target / num_blocks)
        filename_report = f'all_in_one_diff_{len_prompt}_{len_target}_{num_blocks}_abs_per_{type_aggregate}_{type_fn}_{stamp}.json'

        dict_filename_to_list_idx_sorted = defaultdict(list)

        for filename in os.listdir(folder_kv_base):
            if filename[0] == '.':
                continue
            # end

            # matrix_sim_step_layer_token, num_block, len_prompt, size_block, path_kv_base, filename
            list_diff_sorted = self.load_sim_matrix_and_transform_to_most_diff_list_per(
                folder_kv_base,
                filename,
                num_blocks,
                len_prompt,
                size_block,
                type_aggregate=type_aggregate
            )

            dict_filename_to_list_idx_sorted[filename] = list_diff_sorted
        # end

        with open(filename_report, 'w+') as file:
            file.write(json.dumps(dict_filename_to_list_idx_sorted))
        # end

        return self
    # end
# end

