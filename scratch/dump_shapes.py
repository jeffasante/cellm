import json
import struct

def dump_shapes(path):
    with open(path, 'rb') as f:
        magic = f.read(5)
        if magic != b'CELLM':
            print(f"Bad magic: {magic}")
            return
        version = struct.unpack('B', f.read(1))[0]
        header_len = struct.unpack('<I', f.read(4))[0]
        header_json = f.read(header_len).decode('utf-8')
        header = json.loads(header_json)
        for t in header['tensors']:
            if 'self_attn.k_proj.weight' in t['name']:
                print(f"{t['name']}: {t['shape']}")

dump_shapes('models/to-huggingface/gemma-4-E2B-it-int4-aggr-v5/gemma-4-E2B-it-int4-aggr-v5.cellmd')
