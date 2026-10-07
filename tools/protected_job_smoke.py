"""CPU-only lifecycle sentinel; no model training or CUDA context."""
import json
import os
import sys
import time
import torch

print(json.dumps({'event':'START','pid':os.getpid(),'torch':str(torch.__version__),
    'device':'cpu','argv':sys.argv[1:]}),flush=True)
for tick in range(24):
    # Minimal CPU-only work also checks the scheduled user's Python runtime.
    value=float(torch.arange(8,dtype=torch.float32).sum())
    assert value==28
    print(json.dumps({'event':'TICK','tick':tick,'monotonic':time.monotonic()}),flush=True)
    time.sleep(1)
print(json.dumps({'event':'COMPLETE','ticks':24}),flush=True)
