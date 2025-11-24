# inspect_safetensors.py
import os
from safetensors.torch import load_file
import torch
import json

MODEL_DIR = r"C:\Users\Ahas Kaushik\OneDrive\Documents\aiml\5TH SEM_BMSCE\Sign_language_det\hand-sign-detection\models"  # change if your safetensors are elsewhere
OUTFILE = "safetensors_inspect.txt"

def inspect_file(path):
    print(f"\n=== Inspecting: {path} ===")
    try:
        state = load_file(path, device="cpu")
    except Exception as e:
        print("ERROR loading safetensors:", e)
        return {"path": path, "error": str(e)}

    keys = list(state.keys())
    info = []
    for i, k in enumerate(keys[:200]):   # limit to 200 keys so output isn't huge
        try:
            t = state[k]
            # t is a torch.Tensor; show dtype and shape
            info.append({"key": k, "shape": tuple(t.shape), "dtype": str(t.dtype)})
        except Exception as e:
            info.append({"key": k, "shape": "ERR", "dtype": str(e)})
    summary = {
        "path": path,
        "num_keys": len(keys),
        "sample_keys_count": len(info),
        "sample_keys": info
    }
    # print succinct summary
    print("Num keys:", summary["num_keys"])
    print("Sample keys (first 30):")
    for item in info[:30]:
        print(f"  {item['key']}  shape={item['shape']} dtype={item['dtype']}")
    return summary

def main():
    results = []
    if not os.path.isdir(MODEL_DIR):
        print(f"Model dir '{MODEL_DIR}' not found. Please adjust MODEL_DIR variable.")
        return
    files = [f for f in os.listdir(MODEL_DIR) if f.endswith(".safetensors")]
    if not files:
        print("No .safetensors files found in", MODEL_DIR)
        return
    for f in files:
        path = os.path.join(MODEL_DIR, f)
        res = inspect_file(path)
        results.append(res)

    # save JSON summary
    with open(OUTFILE, "w", encoding="utf-8") as fo:
        json.dump(results, fo, indent=2)
    print(f"\nDetailed summary written to {OUTFILE}")

if __name__ == "__main__":
    main()
