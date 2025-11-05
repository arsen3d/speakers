import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from huggingface_hub import login, snapshot_download

# Load environment variables from .env file
load_dotenv()

# Login to Hugging Face if token is available
hf_token = os.getenv('HF_TOKEN')
if hf_token:
    try:
        login(hf_token)
    except Exception as e:
        print(f"Failed to login to Hugging Face: {e}")
        print("Continuing without login...")

# Model configurations
MODELS = [
    {
        "name": "pyannote/speaker-diarization-3.1",
        "local_dir": "data/pyannote/speaker-diarization-3.1",
        "url": "https://huggingface.co/pyannote/speaker-diarization-3.1"
    },
    {
        "name": "pyannote/segmentation-3.0",
        "local_dir": "data/pyannote/segmentation-3.0",
        "url": "https://huggingface.co/pyannote/segmentation-3.0"
    },
    {
        "name": "pyannote/wespeaker-voxceleb-resnet34-LM",
        "local_dir": "data/pyannote/wespeaker-voxceleb-resnet34-LM",
        "url": "https://huggingface.co/pyannote/wespeaker-voxceleb-resnet34-LM"
    }
]

def download_models():
    """Download the required PyAnnote models to the data directory."""
    try:
        # Create data directory if not exists
        os.makedirs("data", exist_ok=True)
        
        print("Starting model downloads...")
        print("=" * 60)
        
        # Download models
        for model in MODELS:
            model_name = model["name"]
            local_dir = model["local_dir"]
            model_url = model["url"]
            
            if os.path.exists(local_dir):
                print(f"✓ {model_name} already exists at {local_dir}")
            else:
                print(f"\n↓ Downloading {model_name}...")
                print(f"  URL: {model_url}")
                try:
                    snapshot_download(
                        model_name,
                        local_dir=local_dir,
                        token=hf_token,
                        repo_type="model"
                    )
                    print(f"✓ Successfully downloaded {model_name}")
                except Exception as e:
                    error_msg = str(e)
                    if "403" in error_msg or "gated" in error_msg.lower() or "forbidden" in error_msg.lower():
                        print(f"✗ Access denied to {model_name}. This model is gated.")
                        print(f"  Please visit {model_url} to request access and accept user conditions.")
                        return False
                    else:
                        print(f"✗ Failed to download {model_name}: {e}")
                        return False
        
        print("\n" + "=" * 60)
        print("✓ All models downloaded successfully!")
        print("=" * 60)
        return True
        
    except Exception as e:
        print(f"\n✗ Error during model download: {e}")
        return False

def verify_models():
    """Verify that all required models are downloaded."""
    all_exist = True
    print("\nVerifying downloaded models...")
    print("-" * 60)
    
    for model in MODELS:
        local_dir = model["local_dir"]
        if os.path.exists(local_dir):
            file_count = len(list(Path(local_dir).rglob("*")))
            print(f"✓ {local_dir}: {file_count} files")
        else:
            print(f"✗ {local_dir}: NOT FOUND")
            all_exist = False
    
    print("-" * 60)
    return all_exist

if __name__ == "__main__":
    success = download_models()
    if success:
        verify_models()
        sys.exit(0)
    else:
        sys.exit(1)