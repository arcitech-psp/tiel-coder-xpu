"""Attach an official BF16 MTP block to an existing compressed checkpoint.

The body checkpoint is kept unchanged. Only tensors whose names begin with
``mtp.`` are read from the BF16 reference and written to a new MTP shard.
The output index marks that block as unquantized.

Usage: python make_bf16_mtp.py BODY BF16_REFERENCE OUTPUT
"""
import json
import os
import shutil

from safetensors import safe_open
from safetensors.torch import save_file

SRC, BF16, DST = __import__('sys').argv[1:4]

os.makedirs(DST, exist_ok=True)
index = json.load(open(f"{SRC}/model.safetensors.index.json"))
wmap = index["weight_map"]

# 1. BF16 MTP tensors from the reference release.
bf16_index = json.load(open(f"{BF16}/model.safetensors.index.json"))
shard = {f for k, f in bf16_index["weight_map"].items() if k.startswith("mtp.")}
assert len(shard) == 1, shard
mtp = {}
with safe_open(f"{BF16}/{shard.pop()}", framework="pt") as f:
    for k in f.keys():
        if k.startswith("mtp."):
            mtp[k] = f.get_tensor(k)
print("bf16 mtp tensors:", len(mtp))

# 2. Drop the quantized MTP entries and keep the remaining body entries unchanged.
dropped = [k for k in wmap if k.startswith("mtp.")]
for k in dropped:
    del wmap[k]
print("dropped int4 mtp entries:", len(dropped))

out_shard = "model-mtp-bf16.safetensors"
save_file(mtp, f"{DST}/{out_shard}", metadata={"format": "pt"})
for k in mtp:
    wmap[k] = out_shard
index["metadata"] = {"total_size": index.get("metadata", {}).get("total_size", 0)}
json.dump(index, open(f"{DST}/model.safetensors.index.json", "w"), indent=1)

# 3. Link body shards and copy metadata files.
for name in os.listdir(SRC):
    if name in ("model.safetensors.index.json", "config.json"):
        continue
    src, dst = f"{SRC}/{name}", f"{DST}/{name}"
    if os.path.lexists(dst):
        continue
    if name.endswith(".safetensors"):
        os.symlink(f"../{os.path.basename(SRC)}/{name}", dst)
    else:
        shutil.copy2(src, dst)

# 4. Mark the MTP block as BF16 for loaders that inspect quantization metadata.
config = json.load(open(f"{SRC}/config.json"))
extra = config["quantization_config"].setdefault("extra_config", {})
for pattern in (".*mtp\\..*", "mtp", ".*mtp\\.layers.*"):
    extra[pattern] = {"bits": 16, "data_type": "float"}
json.dump(config, open(f"{DST}/config.json", "w"), indent=1)
print("wrote", DST)
