import os
import sys
import time

import nbformat
from nbclient import NotebookClient

path = os.path.abspath(sys.argv[1])
root = os.path.dirname(os.path.dirname(path))  # <repo>/examples/x.ipynb -> <repo>
nb = nbformat.read(path, as_version=4)
client = NotebookClient(
    nb,
    timeout=7200,
    kernel_name="python3",
    resources={"metadata": {"path": root}},
)
start = time.time()
client.execute()
nbformat.write(nb, path)
print(f"notebook executed OK in {time.time() - start:.0f}s")
