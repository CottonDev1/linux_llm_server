@echo off
set PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin\x64;C:\Projects\LLM_Website\llamacpp_venv\Lib\site-packages\llama_cpp\lib;%PATH%
C:\Projects\LLM_Website\llamacpp_venv\Scripts\python.exe -c "from llama_cpp import llama_cpp; print('GPU offload supported:', llama_cpp.llama_supports_gpu_offload())"
