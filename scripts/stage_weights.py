#!/usr/bin/env python3
"""Stage safetensors weights into a flat binary blob for Mojo loading.

Reads safetensors file(s), concatenates all tensor data into a single .weights
file, and writes a .offsets text file mapping tensor names to byte offsets.

Usage:
  python stage_weights.py <model_dir> <output_prefix>

Outputs:
  <output_prefix>.weights  - flat binary blob, all tensor data concatenated
  <output_prefix>.offsets  - text file: name\toffset\tsize\tshape per line
"""

import json
import os
import struct
import sys

def read_safetensors_header(filepath):
    with open(filepath, "rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(header_len)), header_len

def main():
    model_dir = sys.argv[1]
    output_prefix = sys.argv[2]
    
    weights_path = output_prefix + ".weights"
    offsets_path = output_prefix + ".offsets"
    
    index_path = os.path.join(model_dir, "model.safetensors.index.json")
    single_file = os.path.join(model_dir, "model.safetensors")
    
    if os.path.exists(index_path):
        with open(index_path) as f:
            index = json.load(f)
        weight_map = index["weight_map"]
        
        # Group by shard
        shard_tensors = {}
        for tensor_name, shard_file in weight_map.items():
            shard_tensors.setdefault(shard_file, []).append(tensor_name)
        
        # Read each shard, extract tensor data
        all_entries = []  # (name, shape, dtype, data_bytes)
        for shard_file in sorted(shard_tensors.keys()):
            shard_path = os.path.join(model_dir, shard_file)
            header, header_len = read_safetensors_header(shard_path)
            data_start = 8 + header_len
            
            with open(shard_path, "rb") as f:
                for tensor_name in shard_tensors[shard_file]:
                    if tensor_name not in header:
                        continue
                    meta = header[tensor_name]
                    off_start, off_end = meta["data_offsets"]
                    f.seek(data_start + off_start)
                    data = f.read(off_end - off_start)
                    all_entries.append((tensor_name, meta["shape"], meta["dtype"], data))
    elif os.path.exists(single_file):
        header, header_len = read_safetensors_header(single_file)
        data_start = 8 + header_len
        
        all_entries = []
        with open(single_file, "rb") as f:
            for name, meta in header.items():
                if name == "__metadata__":
                    continue
                off_start, off_end = meta["data_offsets"]
                f.seek(data_start + off_start)
                data = f.read(off_end - off_start)
                all_entries.append((name, meta["shape"], meta["dtype"], data))
    else:
        print(f"No safetensors files found in {model_dir}")
        sys.exit(1)
    
    # Tied embeddings: 4B checkpoint omits lm_head.weight; tiny_random saves it
    # anyway. Drop it so the Mojo loader sees a uniform tensor set, but only
    # after proving it is byte-identical to embed_tokens (i.e. actually tied).
    by_name = {name: data for name, _, _, data in all_entries}
    if "lm_head.weight" in by_name:
        if by_name["lm_head.weight"] != by_name["model.embed_tokens.weight"]:
            print("FATAL: lm_head.weight present but differs from embed_tokens.weight")
            sys.exit(1)
        all_entries = [e for e in all_entries if e[0] != "lm_head.weight"]
        print("Dropped lm_head.weight (byte-identical to embed_tokens, tied)")

    # Write flat binary blob + offsets
    current_offset = 0
    with open(weights_path, "wb") as wf, open(offsets_path, "w") as of:
        for name, shape, dtype, data in all_entries:
            wf.write(data)
            shape_str = ",".join(str(s) for s in shape)
            of.write(f"{name}\t{current_offset}\t{len(data)}\t{shape_str}\t{dtype}\n")
            current_offset += len(data)
    
    total = current_offset
    print(f"Staged {len(all_entries)} tensors")
    print(f"  Weights: {weights_path} ({total:,} bytes = {total/1e9:.2f} GB)")
    print(f"  Offsets: {offsets_path}")

if __name__ == "__main__":
    main()
