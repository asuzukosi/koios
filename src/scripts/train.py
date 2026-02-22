import argparse, os, sys, pathlib, time
import torch
from src.tokenizer import ByteTokenizer
from src.model import GPT
from src.dataset import ByteDataset


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, required=True, help="Path to the data file")
    p.add_argument("--steps", type=int, required=True, help="Number of steps to train")
    p.add_argument("--sample_every", type=int, required=True, help="Number of steps to sample")
    p.add_argument("--batch_size", type=int, required=True, help="Batch size")
    p.add_argument("--block_size", type=int, required=True, help="Block size")
    p.add_argument("--n_layers", type=int, required=True, help="Number of layers")
    p.add_argument("--n_head", type=int, required=True, help="Number of heads")
    p.add_argument("--d_model", type=int, required=True, help="Model dimension")
    p.add_argument("--dropout", type=float, required=True, help="Dropout rate")
    p.add_argument("--lr", type=float,default=3e-4, help="Learning rate")
    p.add_argument("--weight_decay", type=float,default=0.1, help="Weight decay")
    p.add_argument("--amp", action="store_true",default=False, help="Use automatic mixed precision")
    p.add_argument("--clip", type=float,default=1.0, help="Gradient clip")
    p.add_argument("--compile", action="store_true",default=False, help="Compile the model")
    p.add_argument("--eval_interval", type=int,default=200, help="Evaluation interval")
    p.add_argument("--eval_iters", type=int,default=50, help="Number of iterations to evaluate")

    p.add_argument("--sample_every", type=int,default=200, help="Number of steps to sample from the model")
    p.add_argument("--sample_tokens", type=int,default=256, help="Number of tokens to sample from the model")
    p.add_argument("--temperature", type=float,default=1.0, help="Temperature for sampling")
    p.add_argument("--top_k", type=int,default=40, help="Top-k for sampling")
    p.add_argument("--top_p", type=float,default=0.9, help="Top-p for sampling")
    p.add_argument("--seed", type=int,default=42, help="Seed for sampling")
    p.add_argument("--device", type=str,default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use")
    args = p.parse_args()

    # assert the data file exists
    assert os.path.exists(args.data), "Data file does not exist"
    # assert the steps is a positive integer
    assert args.steps > 0, "Steps must be a positive integer"
    # assert the sample_every is a positive integer
    assert args.sample_every > 0, "Sample every must be a positive integer"
    # assert the batch_size is a positive integer
    assert args.batch_size > 0, "Batch size must be a positive integer"
    # assert the block_size is a positive integer
    assert args.block_size > 0, "Block size must be a positive integer"
    # assert the n_layers is a positive integer
    assert args.n_layers > 0, "Number of layers must be a positive integer"
    # assert the n_head is a positive integer
    assert args.n_head > 0, "Number of heads must be a positive integer"
    # assert the d_model is a positive integer
    assert args.d_model > 0, "Model dimension must be a positive integer"
    # assert the dropout is a float between 0 and 1
    assert 0 <= args.dropout <= 1, "Dropout rate must be between 0 and 1"

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = ByteTokenizer()
    ds = ByteDataset(args.data, block_size=args.block_size)
    model = GPT(tok.vocab_size, args.block_size, args.n_layers, args.n_head, args.n_embed, args.dropout).to(args.device)

    if args.compile and hasattr(torch, "compile"):
        model = torch.compile(model)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=(args.amp and args.device.type == "cuda"))

    best_val = float("inf")
    t0  = time.time()
    model.train()

    for step in range(1, args.steps + 1):
        xb, yb = ds.get_batch('train', args.batch_size, args.device)
        with torch.cuda.amp.autocast(enabled=(args.amp and args.device.type == "cuda")):
            _, loss = model(xb, yb)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        if args.clip > 0:
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        scaler.step(opt)
        scaler.update()

        if step % 50 == 0:
            print(f"step {step:5d} | loss {loss.item():.4f} | {time.time() - t0:.2f}s")
            t0 = time.time()
        if step % args.eval_interval == 0:
            losses = estimate_loss(model, ds, args)
            print(f"step {step:5d} | train loss {losses['train']:.4f} | val loss {losses['val']:.4f}")
            if losses['val'] < best_val:
                best_val = losses['val']
                ckpt_path = args.out_dir / f"model_best.pth"
                os.makedirs(args.out_dir, exist_ok=True)
                torch.save({"model": model.state_dict(), 
                            "config": {
                                "vocab_size": tok.vocab_size, 
                                "block_size": args.block_size,
                                "n_layers": args.n_layers,
                                "n_head": args.n_head,
                                "d_model": args.d_model,
                                "dropout": args.dropout
                            }}, ckpt_path)
        if args.sample_every > 0 and step % args.sample_every == 0:
            start = torch.randint(low=0, high=len(ds.train) - args.block_size -1, size=(1,)).item()
            seed = ds.train[start:start+args.block_size].unsqueeze(0).to(args.device)
            out = model.generate(seed, max_new_tokens=args.sample_tokens, temperature=args.temperature, top_k=args.top_k, top_p=args.top_p)
            txt = tok.decode(out[0].cpu().tolist())
            print(f"step {step:5d} | sample: {txt}")
        
        # final save
        ckpt_path = args.out_dir / f"model_final.pth"
        torch.save({"model": model.state_dict()}, ckpt_path)
        print(f"model saved to {ckpt_path}")

if __name__ == "__main__":
    main()