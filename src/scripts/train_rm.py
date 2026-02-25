from argparse import ArgumentParser
import torch
from src.data_prefs import load_preferences, PrefExample
from src.model import RewardModel
from src.data_prefs import PairCollator
from src.loss import bradley_terry_loss, margin_ranking_loss
from typing import List
from pathlib import Path

def main():
    p = ArgumentParser()
    p.add_argument("--data", type=str, required=True, help="Path to the data file")
    p.add_argument("--steps", type=int, required=True, help="Number of steps to train")
    p.add_argument("--sample_every", type=int, required=True, help="Number of steps to sample")
    p.add_argument("--batch_size", type=int, required=True, help="Batch size")
    p.add_argument("--block_size", type=int, required=True, help="Block size")
    p.add_argument("--n_layers", type=int, required=True, help="Number of layers")
    p.add_argument("--n_head", type=int, required=True, help="Number of heads")
    p.add_argument("--d_model", type=int, required=True, help="Model dimension")
    p.add_argument("--dropout", type=float, required=True, help="Dropout rate")
    p.add_argument("--loss", type=str, required=True, default="bt", help="Loss function")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    # load the data
    data: List[PrefExample] = load_preferences(split='train[:80]')
    # load the model
    triples = [(it.prompt, it.chosen, it.rejected) for it in data]
    # collator + model
    col = PairCollator(block_size=args.block_size, bpe_dir=args.bpe_dir)
    model = RewardModel(vocab_size=col.vocab_size, block_size=args.block_size,
                        n_layer=args.n_layer, n_head=args.n_head, d_model=args.d_model)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
    step = 0; i = 0

    while step < args.steps:
        batch = triples[i:i+args.batch_size]
        if not batch:
            i = 0; continue
        pos, neg = col.collate(batch)
        pos, neg = pos.to(device), neg.to(device)
        r_pos = model(pos)
        r_neg = model(neg)
        if args.loss == "bt":
            loss = bradley_terry_loss(r_pos, r_neg)
        else:
            loss = margin_ranking_loss(r_pos, r_neg, margin=1.0)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        step += 1; i+=args.batch_size
        if step % 25 == 0:
            acc = (r_pos > r_neg).float().mean().item()
            print(f"step {step:04d} | loss {loss.item():.4f} | acc {acc:.4f}")
        Path(args.out).mkdir(parents=True, exist_ok=True)
        torch.save({
            'model': model.state_dict(),
            'config': {
                'vocab_size': col.vocab_size,
                'block_size': args.block_size,
                'n_layers': args.n_layers,
                'n_head': args.n_head,
                'd_model': args.d_model,
                'dropout': args.dropout
            }
        })
        print(f"model saved to {args.out / 'model.pth'}")

if __name__ == "__main__":
    main()


    