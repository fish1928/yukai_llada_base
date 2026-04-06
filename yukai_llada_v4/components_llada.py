import torch

class SimpleLogitsSnapshot:

    def _regularize(self, sample, target):
        return  sample[:, :target.shape[1]]
    # end

    def __init__(self, logits, x, y, id_mask):
        self.id_mask = id_mask

        self.logits = logits

        self.x = self._regularize(x, logits)
        self.y = self._regularize(y, logits)

        self.x0 = torch.argmax(self.logits, dim=-1)

        self.p_finalized = torch.zeros(self.x.shape, dtype=torch.float64).to(self.x.device)
    # end

    def get_x(self):
        return self.x
    # end

    def get_y(self):
        return self.y
    # end

    def get_logits(self):
        return self.logits
    # end

    def get_p_finalized(self):
        return self.p_finalized
    # end

    def transform_logits(self, collector):

        logits_tranform = self.logits
        p = F.softmax(logits_tranform.to(torch.float64), dim=-1)

        index_p_all = collector.get_index(self)

        x0_p = torch.gather(p, dim=-1, index=index_p_all).squeeze(-1)

        neg_inf = torch.tensor(torch.finfo(x0_p.dtype).min, device=x0_p.device, dtype=x0_p.dtype)

        mask_mask = self.x == self.id_mask
        conf = torch.where(mask_mask, x0_p, neg_inf)  # (B, L)   # so only the masked part has confidence

        return conf
    # end

    def materialize_by_idx_(self, idx, conf):

        x0_target = torch.gather(self.x0, dim=-1, index=idx)
        conf_target = torch.gather(conf, dim=-1, index=idx)
        self.x.scatter_(1, idx, x0_target)
        self.p_finalized.scatter_(1, idx, conf_target)
    # end

    def update_logits_(self, idx_transform, logits):
        B, L, H = logits.shape
        assert idx_transform.dim() == 2, "idx_transform.dim(): {} == 2 false".format(idx_transform.dim())
        
        idx_logits = idx_transform.view(B,-1,1).expand(B, -1, H)

        # end match

        self.logits.scatter_(1, idx_logits, logits)
        x0 = torch.argmax(logits, dim=-1)
        self.x0.scatter_(1, idx_transform, x0)
    # end

    def update_this(self, dim, idx_src, idx_tgt=None, **kwargs):

        if idx_tgt is None:
            idx_transform = idx_src
        else:
            idx_tgt=idx_tgt.unsqueeze(0)
            
            idx_transform = torch.gather(idx_tgt, dim=-1, index=idx_src)
        # end

        for k, v in kwargs.items(): # k is a local property name, v is the target to scatter
            v.scatter_(dim, idx_transform, torch.gather(getattr(self, k), dim=dim, index=idx_src))
        # end

        return self
    # end

# end
