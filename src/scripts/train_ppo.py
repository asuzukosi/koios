from argparse import ArgumentParser
import torch
from src.data_prefs import load_preferences, PrefExample
from src.model import RewardModel
from src.data_prefs import PairCollator
from src.loss import bradley_terry_loss, margin_ranking_loss
from typing import List
from pathlib import Path
from src.bpe import RLHFTokenizer
from src.policy import PolicyWithValue
from src.rollout import sample_prompts, model_logprobs, approx_kl
from src.loss import ppo_losses

TEMPLATE = """<|im_start|>user
{prompt}<|im_end|>
<|im_start|>assistant
{response}<|im_end|>
"""

SFT_TEMPLATE = """
<|im_start|>user
{prompt}
<|im_end|>
<|im_start|>assistant
{response}
<|im_end|>
"""





def format_prompt_only(prompt: str) -> str:
    return TEMPLATE.format(prompt=prompt, response="")


def format_example(self, example: tuple[str, str]) -> str:
        """
        format the example (instruction, response) into a string
        """
        return SFT_TEMPLATE.format(prompt=example[0], response=example[1])


def compute_reward(rm: RewardModel, tok: RLHFTokenizer, prompt: str, response: str, device: torch.device) -> float:
    text = format_example((prompt, response))[:-10]
    ids = tok.encode(text)
    tensor = torch.tensor([ids[:tok.block_size]], dtype=torch.long, device=device)
    with torch.no_grad():
        reward = rm(tensor)
    return float(reward[0].item())

def main():
    p = ArgumentParser()
    p.add_argument("--out", type=str, default="out/ppo")
    p.add_argument("--data", type=str, default="data/tiny_hi.txt")
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--sample_every", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--block_size", type=int, default=128)
    p.add_argument("--n_layers", type=int, default=2)
    p.add_argument("--n_head", type=int, default=2)
    p.add_argument("--d_model", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.1)
    args = p.parse_args()
    # get device for the model to run on 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tok = RLHFTokenizer(args.block_size, args.bpe_dir, args.vocab_size)

    ckpt = torch.load(args.policy_ckpt, map_location=device)
    cfg = ckpt.get("config", {})
    vocab_size = cfg.get("vocab_size", tok.vocab_size)
    block_size = cfg.get("block_size", args.block_size)
    n_layer = cfg.get("n_layer", 2)
    n_head = cfg.get("n_head", 2)
    d_model = cfg.get("d_model", 128)
    dropout = cfg.get("dropout", 0.1)

    policy = PolicyWithValue(vocab_size, block_size, n_layer, n_head, d_model).to(device)
    policy.lm.load_state_dict(ckpt.get("model", {}))
    policy.lm.to(device)
    policy.lm.eval()

    ref = PolicyWithValue(vocab_size, block_size, n_layer, n_head, d_model).to(device)
    ref.lm.load_state_dict(ckpt.get("ref", {}))
    for p_ in ref.parameters():
        p_.requires_grad_(False)
    ref.eval()


    # laod reward model
    rckpt = torch.load(args.reward_ckpt, map_location=device)
    rm = RewardModel(vocab_size, block_size, n_layer, n_head, d_model).to(device)
    rm.load_state_dict(rckpt.get("model", {}))
    rm.to(device)
    rm.eval()

    opt = torch.optim.AdamW(policy.parameters(), lr=args.lr, betas=(0.9, 0.999))

    # small prompt pool
    prompts = sample_prompts(16)

    step = 0
    while step < args.steps:
        batch_prompts = prompts[(step*args.batch_size) % len(prompts) : ((step+1)*args.batch_size) % len(prompts)]
        if len(batch_prompts) < args.batch_size:
            batch_prompts += prompts[:args.batch_size - len(batch_prompts)]
        texts = [format_prompt_only(p)[:-10] for p in batch_prompts]
        in_ids = [tok.encode(t) for t in texts]

        with torch.no_grad():
            out_ids = []
            for i, x in enumerate(in_ids):
                idx = torch.tensor([x], dtype=torch.long, device=device)
                out = policy.generate(idx, max_new_tokens=args.resp_len, temperature=args.temperature, top_k=3)
                out_ids.append(out[0].tolist())
        # split prompt/response per sample
        data = []
        for i, prompt in enumerate(batch_prompts):
            full = out_ids[i]
            p_ids = in_ids[i][-block_size:]
            boundary = len(p_ids)
            resp_ids = full[boundary:]
            resp_text = tok.decode(resp_ids)
            r_scalar = compute_reward(rm, tok, prompt, resp_text, device)
            data.append((torch.tensor(full, dtype=torch.long), boundary, r_scalar))
        policy_ctx = getattr(policy, "block_size", block_size)
        max_len = min(policy_ctx, max(t[0].numel() for t in data))
        B = len(data)
        seq = torch.zeros(B, max_len, dtype=torch.long, device=device)
        mask = torch.zeros(B, max_len, dtype=torch.bool, device=device)
        last_idx = torch.zeros(B, dtype=torch.long, device=device)
        rewards = torch.zeros(B, max_len, dtype=torch.float, device=device)

        for i, (ids, boundary, r_scaler) in enumerate(data):
            L_full = ids.numel()
            L = min(L_full, max_len)
            drop = L_full - L
            b = max(0, boundary - drop)
            seq[i, :L] = ids[:L]
            if L < max_len:
                seq[i, L:] = 2
            mask[i, b:L] = True
            rewards[i, L-1] = r_scaler
            last_idx[i] = L-1

        # logprobs & values for policy and reference
        # model logprobs return (B, T-1) for next-token logp; align to seq[:, 1:]
        pol_lp = model_logprobs(policy, seq, mask)
        ref_lp = model_logprobs(ref, seq, mask)
        # values for seq position (B, T)
        with torch.no_grad():
            logits, values, _ = policy(seq, None)
        values = values[:, :-1]
        # select only action positions

        act_mask = mask[:, 1]
        old_logp = pol_lp[act_mask].detach()
        ref_lopg = ref_lp[act_mask].detach()
        old_values = values[act_mask].detach()

        # calculate KL divergence
        kl = (old_logp - ref_lopg)
        shaped_r = rewards[:, 1:][act_mask] - args.kl_coef * kl # penalty for drifting
        
        returns = shaped_r # target value = immediate shaped reward
        adv = returns - old_values
        adv = (adv - adv.mean()) / (adv.std().clamp_min(1e-8))
        # train the policy
        policy.train()
        logits_new, values_new_full, _ = policy(seq, None)
        logp_full = torch.log_softmax(logits_new[:, :-1, :], dim=-1)
        labels = seq[:, 1:]
        new_logp_all = logp_full.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
        new_logp = new_logp_all[act_mask]
        new_values = values_new_full[:, :-1][act_mask]
        out_loss = ppo_losses(new_logp, old_logp, new_values, old_values, returns, clip_ratio=0.2, vf_coef=0.5, ent_coef=0.0)
        loss = out_loss.total_loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        opt.step()
        policy.eval()
        # update the reference
        with torch.no_grad():
           lp_post = model_logprobs(policy, seq, mask)
           lp_post = lp_post[act_mask]
           kl_post = (old_logp - lp_post).mean()
           
           lp_now  = lp_post
           kl_ref_now = (lp_now - ref_lopg).mean()
        step += 1
        if step % args.sample_every == 0:
            print(f"Step {step} - Loss: {loss.item():.4f} - KL: {kl_post.item():.4f} - KL Ref: {kl_ref_now.item():.4f}")

    Path(args.out).mkdir(parents=True, exist_ok=True)
    torch.save({
        "policy": policy.state_dict(),
        "ref": ref.state_dict(),
        "rm": rm.state_dict(),
        "opt": opt.state_dict(),
        "step": step,
    }, Path(args.out) / "checkpoint.pt")


if __name__ == "__main__":
    main()







