import torch
from argparse import ArgumentParser
from src.policy import PolicyWithValue
from src.model import RewardModel
from src.bpe import RLHFTokenizer
from src.rollout import sample_prompts
from src.scripts.train_ppo import format_prompt_only, format_example

def score_policy(policy_ckpt: str, reward_ckpt: str, split: str, bpe_dir: str, device: torch.device) -> float:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = RLHFTokenizer(block_size=256, bpe_dir=bpe_dir)
    ckpt = torch.load(policy_ckpt, map_location=device)
    cfg = ckpt.get("config", {})
    vocab_size = cfg.get("vocab_size", tok.vocab_size)
    block_size = cfg.get("block_size", 256)
    n_layer = cfg.get("n_layer", 2)
    n_head = cfg.get("n_head", 2)
    d_model = cfg.get("d_model", 128)
    dropout = cfg.get("dropout", 0.1)
    policy = PolicyWithValue(vocab_size, block_size, n_layer, n_head, d_model).to(device)
    policy.load_state_dict(ckpt.get("policy", {}))
    policy.eval()
    # for comparing against reference policy
    ref = PolicyWithValue(vocab_size, block_size, n_layer, n_head, d_model).to(device)
    ckpt_ref = torch.load(policy_ckpt, map_location=device)
    ref.lm.load_state_dict(ckpt_ref.get("model", {}))
    for p_ in ref.parameters():
        p_.requires_grad_(False)
    ref.eval()
    # load reward model
    rckpt = torch.load(reward_ckpt, map_location=device)
    rm = RewardModel(vocab_size, block_size, n_layer, n_head, d_model).to(device)
    rm.load_state_dict(rckpt.get("model", {}))
    rm.to(device)
    rm.eval()

    prompts = sample_prompts(16)
    rewards = []
    for p in prompts:
        prefix = format_prompt_only(p)[:-10]
        ids = tok.encode(prefix)
        idx = torch.tensor([ids], dtype=torch.long, device=device)
        x = torch.tensor([ids[-tok.block_size:]], dtype=torch.long, device=device)
        with torch.no_grad():
            y = policy.generate(x, max_new_tokens=256, temperature=0.0, top_k=3)
            y_old = ref.generate(x, max_new_tokens=256, temperature=0.0, top_k=3)
        resp = tok.decode(y[0].tolist()[len(ids[-tok.block_size:]):])
        resp_old = tok.decode(y_old[0].tolist()[len(ids[-tok.block_size:]):])

        text = format_example((p, resp))[:-10]
        z = torch.tensor([tok.encode(text)], dtype=torch.long, device=device)
        with torch.no_grad():
            r = rm(z)[0].item()
        rewards.append(r)
    return sum(rewards) / max(1, len(rewards)) # the max is to avoid division by zero




def main():
    p = ArgumentParser()
    p.add_argument("--policy_ckpt", type=str, required=True, help="Path to the policy checkpoint file")
    p.add_argument("--reward_ckpt", type=str, required=True, help="Path to the reward checkpoint file")
    p.add_argument("--split", type=str, required=True, help="Split to evaluate")
    p.add_argument("--bpe_dir", type=str, required=True, help="Path to the BPE directory")
    args = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    avg_r = score_policy(args.policy_ckpt, args.reward_ckpt, args.split, args.bpe_dir, device)
    print(f"Average reward: {avg_r:.4f}")

if __name__ == "__main__":
    main()