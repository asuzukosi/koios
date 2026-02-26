import torch, torch.nn.functional as F

def bradley_terry_loss(r_pos: torch.Tensor, r_neg: torch.Tensor) -> torch.Tensor:
    diff = r_pos - r_neg
    return F.softplus(-diff).mean()

def margin_ranking_loss(r_pos: torch.Tensor, r_neg: torch.Tensor, margin: float = 1.0) -> torch.Tensor:
    y = torch.ones_like(r_pos)
    return F.margin_ranking_loss(r_pos, r_neg, y, margin)


class PPOLossOut:
    policy_loss: torch.Tensor
    value_loss: torch.Tensor
    entropy: torch.Tensor
    approx_kl: torch.Tensor
    total: torch.Tensor


def ppo_losses(new_logp: torch.Tensor, old_logp: torch.Tensor, adv: torch.Tensor, new_values: torch.Tensor, old_values: torch.Tensor, returns: torch.Tensor, clip_ratio: float = 0.2, vf_coef: float = 0.5, ent_coef: float = 0.0) -> torch.Tensor:
    ratio = torch.exp(new_logp - old_logp)
    unclipped = ratio * adv
    clipped = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * adv
    policy_loss = -torch.mean(torch.min(unclipped, clipped))
    # value 
    value_loss = F.mse_loss(new_values, returns)
    # entropy bonus
    entropy = -new_logp.mean()
    # approx kl for logging
    approx_kl = torch.mean(old_logp - new_logp)
    total = policy_loss + vf_coef * value_loss + ent_coef * entropy
    return PPOLossOut(policy_loss, value_loss, entropy, approx_kl, total)