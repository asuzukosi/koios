from dataclasses import dataclass
from datasets import load_dataset

@dataclass
class SFTData:
    """
    holds single instance of sft data
    """
    prompt: str
    response: str

# NOTE: train on tiny dataset, make sure the loss is going down before scaling up
def load_sft_hf(split: str="train[:200]") -> list[SFTData]:
    """
    load sft data from huggingface datasets
    """
    output: list[SFTData] = []
    ds = load_dataset("tatsu-lab/alpaca", split=split)
    for row in ds:
        instr = row.get("instruction", "").strip() # system prompt
        inp = row.get("input", "").strip() # input prompt
        out = row.get("output", "").strip() # output prompt
        if inp:
            instr = f"{instr}\n\n{inp}"
        if instr and out:
            output.append(SFTData(prompt=instr, response=out))
    return output