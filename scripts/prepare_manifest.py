#!/usr/bin/env python3
import json, os, struct, sys, glob

DTYPE_MAP = {"BF16": 0, "F16": 1, "F32": 2, "F64": 3, "I32": 4, "I64": 5}

def read_safetensors_header(filepath):
    with open(filepath, "rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(header_len))

def main():
    model_dir = sys.argv[1]
    output_path = sys.argv[2]
    index_path = os.path.join(model_dir, "model.safetensors.index.json")
    single_file = os.path.join(model_dir, "model.safetensors")

    if os.path.exists(index_path):
        with open(index_path) as f:
            index = json.load(f)
        weight_map = index["weight_map"]
        shards = {}
        for tensor_name, shard_file in weight_map.items():
            shards.setdefault(shard_file, {})[tensor_name] = None
        all_tensors = []
        for shard_file in sorted(shards.keys()):
            shard_path = os.path.join(model_dir, shard_file)
            header = read_safetensors_header(shard_path)
            for tensor_name in shards[shard_file]:
                if tensor_name in header:
                    all_tensors.append((tensor_name, header[tensor_name], shard_file))
    elif os.path.exists(single_file):
        header = read_safetensors_header(single_file)
        all_tensors = [(name, meta, "model.safetensors") for name, meta in header.items() if name != "__metadata__"]
    else:
        print(f"No safetensors files found in {model_dir}")
        sys.exit(1)

    with open(output_path, "wb") as f:
        f.write(struct.pack("<I", len(all_tensors)))
        for tensor_name, meta, shard_file in all_tensors:
            name_bytes = tensor_name.encode("utf-8")
            dtype_code = DTYPE_MAP.get(meta["dtype"], 255)
            shape = meta["shape"]
            offsets = meta["data_offsets"]
            f.write(struct.pack("<I", len(name_bytes)))
            f.write(name_bytes)
            f.write(struct.pack("<B", dtype_code))
            f.write(struct.pack("<B", len(shape)))
            for dim in shape:
                f.write(struct.pack("<I", dim))
            f.write(struct.pack("<Q", offsets[0]))
            f.write(struct.pack("<Q", offsets[1] - offsets[0]))

    shard_list_path = output_path + ".shards"
    shard_files = sorted(set(t[2] for t in all_tensors))
    with open(shard_list_path, "w") as f:
        for shard_file in shard_files:
            f.write(os.path.join(model_dir, shard_file) + "\n")

    print(f"Manifest written: {output_path}")
    print(f"  Tensors: {len(all_tensors)}")
    print(f"  Shards: {len(shard_files)}")
    total_bytes = sum(t[1]["data_offsets"][1] - t[1]["data_offsets"][0] for t in all_tensors)
    print(f"  Total weight bytes: {total_bytes:,} ({total_bytes / 1e9:.2f} GB)")

if __name__ == "__main__":
    main()
