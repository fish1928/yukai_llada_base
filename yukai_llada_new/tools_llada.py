import torch
import torch.nn.functional as F
from abc import ABC, abstractmethod
from tools_debug import jprint


class DiffusionQuotaHelper(ABC):
    @abstractmethod
    def get_quota(self, step_current):
        pass
    # end
# end

class BlockDiffusionQuotaHelper(DiffusionQuotaHelper):
    def __init__(self, block_mask_index: torch.Tensor, steps_per_block: int) -> torch.Tensor:
        device = block_mask_index.device
        dtype = torch.long

        total = block_mask_index.sum(dim=1)                  # (B,)
        base  = torch.div(total, steps_per_block, rounding_mode='floor')  # (B,)
        rem   = total - base * steps_per_block                         # (B,)

        # Start with base for all steps
        num_transfer_tokens = base.unsqueeze(1).expand(-1, steps_per_block).to(dtype)  # (B, steps)

        # Add +1 to the first `rem[b]` steps for each batch b — without tensor slicing
        cols = torch.arange(steps_per_block, device=device).unsqueeze(0)               # (1, steps)
        add_mask = cols < rem.unsqueeze(1)                                   # (B, steps)
        self.num_transfer_tokens = num_transfer_tokens + add_mask.to(dtype)       # (B, steps)
    # end

    def get_quota(self, step_current):
        quota_current = self.num_transfer_tokens[:, step_current]

        if quota_current.dim() == 2 and quota_current.size(1) == 1:
            quota_current = quota_current.squeeze(1)
        # end

        return quota_current
    # end
# end


class ConfKSorter:

    def argsort(self, conf_all):
        idx_sorted = torch.argsort(conf_all, dim=1, descending=True)
        jprint('idx_sorted: {}'.format(idx_sorted.shape))
        return idx_sorted
    # end
# end

class RandomKSorter(ConfKSorter):
    def argsort(self, confidence, snapshot):

        confidence = torch.where(
                snapshot.mask_mask,
                torch.rand(confidence.shape[0], confidence.shape[1], device=confidence.device),
                confidence
            )

        return super().argsort(confidence)

    # end
# end


class TopKSorter(ConfKSorter):
    def argsort(self, confidence, snapshot):
        jprint('confidence: {}'.format(confidence.shape))
        return super().argsort(confidence)
    # end
# end


class ConfCollectorInterface(ABC):
    @abstractmethod
    def get_index(self, snapshot):
        pass
    # end
# end

class TruthCollector(ConfCollectorInterface):
    def get_index(self, snapshot):
        index = snapshot.y.unsqueeze(-1)
        return index
    # end
# end


class MaxCollector(ConfCollectorInterface):
    def get_index(self, snapshot):
        index = snapshot.x0
        return index
    # end
# end



class LogitsTransformer:
    def transform_logits(self, logits, collector):
        p = F.softmax(logits.to(torch.float64), dim=-1)
        x0_p = collector.gather_x0_p(p, self)
        return x0_p
    # end
# end