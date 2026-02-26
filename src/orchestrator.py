import subprocess, sys, pathlib, argparse, shlex

ROOT = pathlib.Path(__file__).resolve().parent
OUT = ROOT / "output"
RUN = ROOT / "run"
def run(cmd: str):
    print(f"Running: {cmd}")
    res = subprocess.run(shlex.split(cmd), capture_output=True, text=True)
    if res.returncode != 0:
        print(f"Error: {res.stderr}")
        sys.exit(1)
    return res.stdout


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--visualize", action="store_true", help="Visualize the model")
    p.add_argument("--demo", action="store_true", help="Run the demonstrations")
    args = p.parse_args()
    
    OUT.mkdir(parents=True, exist_ok=True)

    # run sanity check
    run(f"python -m src.demos.attn_numpy")
    # run unit tests
    run("python -m pytest -q tests/test_attn_math.py")
    run("python -m pytest -q tests/test_causal_math.py")
    run("python -m pytest -q tests/test_kvcache_shapes.py")
    run("python -m pytest -q tests/test_rmsnorm.py")
    run("python -m pytest -q tests/test_rope_apply.py")
    run("python -m pytest -q tests/test_tokenizer_bpe.py")
    run("python -m pytest -q tests/test_scheduler.py")
    run("python -m pytest -q tests/test_resume_shapes.py")
    run("ptyhon -m pytest -q tests/test_formatter.py")
    run("python -m pytest -q tests/test_masking.py")
    # matrix math walthrough for mha
    run("python -m src.demos.mha_shapes")

    # quick stroke training on a tiny file path tiny_hi.txt adjust as needed
    run("python src.scripts.train --data data/tiny_hi.txt --steps 400 --sample_every 100 --batch_size 32 --block_size 128 --n_layers 2 --n_head 2 --d_model 128 --dropout 0.1") # block size is the context window size
    # sample from the base checkpoint
    run(f"python src.scripts.sample --checkpoint {RUN / "model.pth"} --tokens 100 --prompt 'Once upon a time'")
    # quick stroke training on a tiny file path tiny_hi.txt adjust as needed
    run("python src.scripts.train_sft --data data/tiny_hi.txt --steps 400 --sample_every 100 --batch_size 32 --block_size 128 --n_layers 2 --n_head 2 --d_model 128 --dropout 0.1")
    run(f"python src.scripts.sample_sft --checkpoint {RUN / "model.pth"} --block_size 256 --tokens 100 --prompt 'Once upon a time'")
    # train the reward model
    run("python src.scripts.train_rm --data data/tiny_hi.txt --steps 400 --sample_every 100 --batch_size 32 --block_size 128 --n_layers 2 --n_head 2 --d_model 128 --dropout 0.1")
    # evaluate the reward model
    run(f"python src.scripts.eval_rm --checkpoint {RUN / "model.pth"} --data data/tiny_hi.txt --iters 50 --block_size 128")
    # rlhf training
    run("python src.scripts.train_ppo --data data/tiny_hi.txt --steps 400 --sample_every 100 --batch_size 32 --block_size 128 --n_layers 2 --n_head 2 --d_model 128 --dropout 0.1")
    # rlhf evaluation
    run(f"python src.scripts.eval_ppo --checkpoint {RUN / "model.pth"} --data data/tiny_hi.txt --iters 50 --block_size 128")
    # evaluate final val loss
    run(f"python src.scripts.evaluate --checkpoint {RUN / "model.pth"} --data data/tiny_hi.txt --iters 50 --block_size 128") # block size is the context window size

    if args.visualize:
        run("python -m src.demos.visualize_multi_head")
        print(f"vizialization images saved to {OUT / 'visualization'}")

    if args.demo:
        run("python -m src.demos.generate --rmsnorm --rope --swiglu --sliding_window 64 --sink 4 --tokens 200")
    
    print(f"all demonstrations completed")


if __name__ == "__main__":
    main()