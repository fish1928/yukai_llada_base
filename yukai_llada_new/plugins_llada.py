import os
import inspect
from abc import ABC, abstractmethod

import torch
import torch.nn.functional as F

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

    def collect_kv_previous_and_calculate_sim_per_step_(self):
        pass
    # end

    def aggregate_result(self):
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
    # end

    def aggregate_result(self):
        for name_hidden in self._get_names_hidden():
            matrixs_sim_per_step = self.dict_hidden_to_matrixs_sim_per_step[name_hidden]
            matrix_sim_per_step_layer_token = torch.stack(matrixs_sim_per_step, 0)  # dimension
            return matrix_sim_per_step_layer_token.detach().float().cpu()
        # end
    # end

    '''further calculation'''
    def token_nonsimilarity_score_abs_per_step(
        self,
        sim: torch.Tensor,
        p: float = 3.0,
        type_fn: str = 'p'
    ) -> torch.Tensor:

        assert sim.ndim == 3, f"Expected 3D tensor [steps, layers, tokens], got shape {tuple(sim.shape)}"

        diff = torch.abs(1.0 - sim)
        if type_fn == 'p':
            score = diff.pow(p).mean(dim=(1)).pow(1.0 / p)
        elif type_fn == 'log':
            score = torch.log1p(diff).mean(dim=(1))
        # end

        return score
    # end

    def load_sim_matrix_and_transform_to_most_diff_list_per_step(self, folder_kv_base, filename, num_block, len_prompt, size_block):
        path_kv_file = os.path.join(folder_kv_base, filename)
        matrix_sim_step_layer_token = torch.load(path_kv_file)
        matrix_sim_step_layer_token = F.pad(matrix_sim_step_layer_token, (0,0,0,0,1,0), value=1.0)

        list_idx_sim_sorted = []

        for id_block in range(num_block):
            position_start = len_prompt + size_block * id_block
            matrix_sim_step_layer_token_cached = matrix_sim_step_layer_token[:size_block, :, :position_start]  #(steps_block, len_cached)
            matrix_step_scores_diff_token = self.token_nonsimilarity_score_abs_per_step(matrix_sim_step_layer_token_cached)
            matrix_step_idx_diff_token_decending = torch.argsort(matrix_step_scores_diff_token, dim=-1, descending=True)

            for step in range(matrix_step_idx_diff_token_decending.shape[0]):
                idxs_diff_token_decending = matrix_step_idx_diff_token_decending[step,:]
                list_idx_sim_sorted.append({'filename': filename, 'block': id_block, 'step': step, 'idx': idxs_diff_token_decending.tolist(), 'value_raw': matrix_step_scores_diff_token[step,:].tolist()})
            # end
        # end

        return list_idx_sim_sorted
    # end

    def token_nonsimilarity_score_abs_per_block(
        self,
        sim: torch.Tensor,
        p: float = 3.0,
        type_fn: str = 'p'
    ) -> torch.Tensor:

        assert sim.ndim == 3, f"Expected 3D tensor [steps, layers, tokens], got shape {tuple(sim.shape)}"

        diff = torch.abs(1.0 - sim)
        if type_fn == 'p':
            score = diff.pow(p).mean(dim=(0, 1)).pow(1.0 / p)
        elif type_fn == 'log':
            score = torch.log1p(diff).mean(dim=(0, 1))
        # end

        return score
    # end

    def load_sim_matrix_and_transform_to_most_diff_list_per_block(self, folder_kv_base, filename, num_block, len_prompt, size_block):
        path_kv_file = os.path.join(folder_kv_base, filename)
        matrix_sim_step_layer_token = torch.load(path_kv_file)
        matrix_sim_step_layer_token = F.pad(matrix_sim_step_layer_token, (0,0,0,0,1,0), value=1.0)

        list_idx_sim_sorted = []

        for id_block in range(num_block):
            position_start = len_prompt + size_block * id_block
            matrix_sim_step_layer_token_cached = matrix_sim_step_layer_token[:size_block, :, :position_start]  #(steps_block, len_cached)
            matrix_step_scores_diff_token = self.token_nonsimilarity_score_abs_per_step(matrix_sim_step_layer_token_cached)
            matrix_step_idx_diff_token_decending = torch.argsort(matrix_step_scores_diff_token, dim=-1, descending=True)

            for step in range(matrix_step_idx_diff_token_decending.shape[0]):
                idxs_diff_token_decending = matrix_step_idx_diff_token_decending[step,:]

                list_idx_sim_sorted.append({'filename': filename, 'block': id_block, 'step': step, 'idx': idxs_diff_token_decending.tolist(), 'value_raw': matrix_step_scores_diff_token[step,:].tolist()})
            # end
        # end

        return list_idx_sim_sorted
    # end


    def dump_all_in_one(self):  # from test_get_top_change.ipynb
        folder_kv_base = 'sims_kv_0315'
        type_fn = 'p'
        filename_report = f'all_in_one_sim_report_abs_per_step_{type_fn}_0315.json'

        len_prompt = 512
        num_block = 8
        len_target = 1024
        size_block = int(len_target / num_block)

        dict_filename_to_list_idx_sorted = defaultdict(list)

        for filename in os.listdir(folder_kv_base):
            if filename[0] == '.':
                continue
            # end

            # matrix_sim_step_layer_token, num_block, len_prompt, size_block, path_kv_base, filename
            list_diff_sorted = self.load_sim_matrix_and_transform_to_most_diff_list(folder_kv_base, filename, num_block, len_prompt, size_block)
            dict_filename_to_list_idx_sorted[filename] = list_diff_sorted
        # end

        with open(filename_report, 'w+') as file:
            file.write(json.dumps(dict_filename_to_list_idx_sorted))
        # end

    # end
# end

