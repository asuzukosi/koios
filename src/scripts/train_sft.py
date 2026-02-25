from argparse import ArgumentParser
import torch
from src.dataset_sft import load_sft_hf
from typing import List
from src.dataset_sft import SFTData
from src.model import GPTModern
from src.curriculum import LengthCurriculum
from src.sft_collator import SFTCollator



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

    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # load the sft data
    data: List[SFTData] = load_sft_hf()

    # print first few examples
    for i, d in enumerate(data[:5]):
        print(f"EXAMPLE {i+1}:")
        print(f"PROMPT: {d.prompt}")
        print(f"RESPONSE: {d.response}")
        print("-" * 50)

    tuples = [(it.prompt, it.response) for it in data]
    curr = list(LengthCurriculum(tuples))

    print(curr)

    # collator + model
    col = SFTCollator(block_size=args.block_size, bpe_dir=args.bpe_dir)
    model = GPTModern(vocab_size=args.vocab_size, block_size=args.block_size, n_layers=args.n_layers, 
                      n_head=args.n_head, d_model=args.d_model, dropout=args.dropout)
    if args.ckpt:
        print(f"using model config from checkpoint {args.ckpt}")
        ckpt = torch.load(args.ckpt, map_location=device)
        cfg = ckpt.get("config", {})
        model.load_state_dict(ckpt.get("model", {}))
    
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    model.train()


