import os
import sys

# Add CUDA to PATH
cuda_bin = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin"
os.environ["PATH"] = cuda_bin + ";" + os.environ.get("PATH", "")

from llama_cpp import llama_cpp
print("GPU offload supported:", llama_cpp.llama_supports_gpu_offload())
