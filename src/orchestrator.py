import subprocess, sys, pathlib, argparse, shlex

ROOT = pathlib.Path(__file__).resolve().parent
OUT = ROOT / "output"

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
    args = p.parse_args()
    
    OUT.mkdir(parents=True, exist_ok=True)

    # run sanity check
    run(f"python -m src.demos.attn_numpy")
    # run unit tests
    # run("python -m pytest -q tests/test_attn_math.py")
    # run("python -m pytest -q tests/test_causal_math.py")
    # matrix math walthrough for mha
    run("python -m src.demos.mha_shapes")

    if args.visualize:
        run("python -m src.demos.visualize_multi_head")
        print(f"vizialization images saved to {OUT / 'visualization'}")
    
    print(f"all demonstrations completed")


if __name__ == "__main__":
    main()