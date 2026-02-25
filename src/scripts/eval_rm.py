from src.model import RewardModel
from argparse import ArgumentParser
import torch
from src.data_prefs import load_preferences
from src.data_prefs import PairCollator
import math

def main():
    p = ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--split", type=str, default="val[:200]")
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--bpe_dir", type=str, default=None)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    items = load_preferences(split=args.split)
    triples = [(it.prompt, it.chosen, it.rejected) for it in items]

    col = PairCollator(block_size=256, bpe_dir=args.bpe_dir)
    ckpt = torch.load(args.ckpt, map_location=device)
    cfg = ckpt.get("config", {})
    model = RewardModel(vocab_size=cfg.get("vocab_size", 32000), block_size=cfg.get("block_size", 256), n_layers=cfg.get("n_layers", 2), 
                        n_head=cfg.get("n_head", 2), d_model=cfg.get("d_model", 128), dropout=cfg.get("dropout", 0.1))
    model.load_state_dict(ckpt.get("model", {}))
    model.to(device)
    model.eval()

    B = 16
    correct = 0; total = 0
    for i in range(0, len(triples), B):
        batch = triples[i: i+B]
        pos, neg = col.collate(batch)
        pos, neg = pos.to(device), neg.to(device)
        with torch.no_grad():
            r_pos = model(pos)
            r_neg = model(neg)
        correct += (r_pos > r_neg).sum().item()
        total += pos.size(0)
    acc = correct / max(1, total)
    print(f"pairs-{total:04d} | acc {acc:.4f}")

if __name__ == "__main__":
    main()