import torch


def _forward_process(mask_id, batch, prompt_index):
    b, l = batch.shape

    target_len = (l - prompt_index.sum()).item()
    k = torch.randint(1, target_len + 1, (), device=batch.device)

    x = torch.round(torch.linspace(float(k), k + (b - 1) * (target_len / b), steps=b, device=batch.device)).long()
    x = ((x - 1) % target_len) + 1
    assert x.min() >= 1 and x.max() <= target_len

    indices = torch.arange(target_len, device=batch.device).repeat(b, 1)
    is_mask = indices < x.unsqueeze(1)

    for i in range(b):
        is_mask[i] = is_mask[i][torch.randperm(target_len)]

    is_mask = torch.cat((torch.zeros(b, prompt_index.sum(), dtype=torch.bool, device=batch.device), is_mask), dim=1)

    noisy_batch = torch.where(is_mask, mask_id, batch)

    return noisy_batch, (x / target_len).unsqueeze(1).repeat(1, l)


@torch.no_grad()
def get_logits(model, cfg, mask_id, batch, prompt_index):
    if cfg > 0.:
        assert len(prompt_index) == batch.shape[1]
        prompt_index = prompt_index.unsqueeze(0).repeat(batch.shape[0], 1)
        un_batch = batch.clone()
        un_batch[prompt_index] = mask_id
        batch = torch.cat([batch, un_batch])

    logits = model(batch).logits

    if cfg > 0.:
        logits, un_logits = torch.chunk(logits, 2, dim=0)
        logits = un_logits + (cfg + 1) * (logits - un_logits)
    return logits[:, :batch.shape[1]]
# end


@torch.no_grad()
def get_loglikelihood(mc_num, batch_size, mask_id, device, prefix, target):
    seq = torch.concatenate([prefix, target])[None, :]
    seq = seq.repeat((batch_size, 1)).to(device)

    prompt_index = torch.arange(seq.shape[1], device=device) < len(prefix)

    loss_acc = []
    for _ in range(mc_num // batch_size):
        perturbed_seq, p_mask = _forward_process(seq, prompt_index)

        mask_indices = perturbed_seq == mask_id

        logits = get_logits(perturbed_seq, prompt_index)

        loss = F.cross_entropy(logits[mask_indices], seq[mask_indices], reduction='none') / p_mask[mask_indices]
        loss = loss.sum() / batch_size
        loss_acc.append(loss.item())

    return - sum(loss_acc) / len(loss_acc)
# end


