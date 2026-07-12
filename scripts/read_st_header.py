import struct, json, glob, os
base = "/mnt/models/models/qwen3-4b-instruct-2507"
files = sorted(glob.glob(base + "/*.safetensors"))
names = [os.path.basename(f) for f in files]
print("Files:", names)
with open(files[0], "rb") as f:
    header_len = struct.unpack("<Q", f.read(8))[0]
    header = json.loads(f.read(header_len))
for k, v in sorted(header.items()):
    if k == "__metadata__":
        print("metadata:", v)
        continue
    print(k, v)
