import json
import struct
import sys

def dump_dtypes(path):
    with open(path, 'rb') as f:
        magic = f.read(5)
        if magic != b'CELLM':
            print(f"Bad magic: {magic}")
            return
        version = struct.unpack('B', f.read(1))[0]
        header_len = struct.unpack('<I', f.read(4))[0]
        header_json = f.read(header_len).decode('utf-8')
        header = json.loads(header_json)
        print(f"Model: {path}")
        print(f"Version: {version}")
        print(f"{'Tensor Name':<60} {'Shape':<20} {'Dtype'}")
        print("-" * 90)
        for t in header['tensors']:
            # Only print first few layers if there are many, plus some key ones
            name = t['name']
            if 'layers.0' in name or 'layers.1' in name or 'norm' in name or 'embed' in name or 'lm_head' in name:
                print(f"{t['name']:<60} {str(t['shape']):<20} {t['dtype']}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        dump_dtypes(sys.argv[1])
    else:
        # Default to the qwen model from the user request
        dump_dtypes('models/to-huggingface/qwen2.5-0.5b-int8-v1/qwen2.5-0.5b-int8-v1.cellm')
