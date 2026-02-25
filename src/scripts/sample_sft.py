from argparse import ArgumentParser
import torch
from src.model import GPTModern
from src.sft_collator import SFTCollator


def main():
    p = ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True, help="Path to the checkpoint file")
    p.add_argument("--block_size", type=int, required=True, help="Block size")
    p.add_argument("--tokens", type=int, required=True, help="Number of tokens to sample")
    p.add_argument("--prompt", type=str, required=True, help="Prompt to sample from")
    p.add_argument("--n_layers", type=int, required=True, help="Number of layers")
    p.add_argument("--n_head", type=int, required=True, help="Number of heads")
    p.add_argument("--d_model", type=int, required=True, help="Model dimension")
    p.add_argument("--dropout", type=float, required=True, help="Dropout rate")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device)
    cfg = ckpt.get("config", {})

    col = SFTCollator(block_size=cfg.get("block_size", 256), bpe_dir=cfg.get("bpe_dir", "data/bpe"))
    model = GPTModern(vocab_size=cfg.get("vocab_size", 32000), block_size=cfg.get("block_size", 256), n_layers=cfg.get("n_layers", 2), 
                      n_head=cfg.get("n_head", 2), d_model=cfg.get("d_model", 128), dropout=cfg.get("dropout", 0.1))
    model.load_state_dict(ckpt.get("model", {}))
    model.to(device)
    model.eval()

    # load the curriculum
    prompt_text =  SFTCollator.format_prompt_only(args.prompt)[:-10]
    ids = col.tok.encode(prompt_text)
    idx = torch.tensor(ids, device=device).unsqueeze(0)

    with torch.no_grad():
        out = model.generate(idx, max_new_tokens=args.tokens)

    # decode 
    output_ids =  out[0].cpu().tolist()
    orig_len = idx.size(1)
    response_text = col.tok.decode(output_ids)
    print(f"response: {response_text}")