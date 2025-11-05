# Idea
Build a python app based on https://github.com/pyannote/pyannote-audio
Be sure to use gradio
use python uv command to install requirements
instead of uv run pip install do uv pip install

# PyTorch Installation
To install PyTorch with CUDA support:
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# User Experience
Select mp3 or wave file.
upload it.
result should be a time stamps of when each speaker is speaking.
