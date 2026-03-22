import inspect
from abc import ABC, abstractmethod


class InspectorPlugin(ABC):

    @abstractmethod
    def get_plugin_name(self):
        raise NotImplementedError
    # end

    def load_vars(self, *args):
        frame = inspect.currentframe().f_back.f_back
        vars_caller = frame.f_locals
        return tuple(vars_caller[arg] for arg in args)
    # end

    def load_func(self, arg):
        frame = inspect.currentframe().f_back.f_back
        vars_caller = frame.f_locals
        self_caller = vars_caller['self']
        return getattr(self_caller, arg)
    # end

    def save_vars(self, **kvargs):
        frame = inspect.currentframe().f_back.f_back
        vars_caller = frame.f_locals
        self_caller = vars_caller['self']
        for k, v in kvargs.items():
            self_caller[k] = v
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
        layer_past, k_current, v_current = self.load_vars('layer_past', 'k_current', 'v_current')

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
        self.save_vars(layer_past=layer_past)
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
       self.save_vars(_k_previous=None, _v_previous=None)
    # end

    def save(self):
        k, v = self.load_vars('k','v')
        self.save_vars(_k_previous=k, _v_previous=v)
    # end

# end