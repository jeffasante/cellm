import struct
import numpy as np
import sys

file = sys.argv[1]
offset = int(sys.argv[2])
count = int(sys.argv[3])

with open(file, 'rb') as f:
    f.seek(offset)
    data = f.read(count * 2)
    # Use numpy to decode half precision
    weights = np.frombuffer(data, dtype=np.float16)
    print(list(weights))
