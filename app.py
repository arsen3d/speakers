#when launching this app, use uv command
import gradio as gr
import os
from dotenv import load_dotenv
import numpy as np
import torch
from huggingface_hub import login
from pyannote.audio import Pipeline
from pyannote.audio.pipelines import SpeakerDiarization
import soundfile as sf
from pydub import AudioSegment
import yaml
import sys

# Re-enable TF32 for better performance
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Load environment variables from .env file
load_dotenv()

# Set cache directory for models (use local data directory)
os.environ['PYANNOTE_CACHE'] = os.path.abspath('data')

# Login to Hugging Face if token is available
hf_token = os.getenv('HF_TOKEN')
if hf_token:
    login(hf_token)

# Lazy-loaded pipeline
pipeline = None
device = None
print(f"CUDA available: {torch.cuda.is_available()}")

def check_models_exist():
    """Check if all required models are downloaded."""
    required_models = [
        "data/pyannote/speaker-diarization-3.1",
        "data/pyannote/segmentation-3.0",
        "data/pyannote/wespeaker-voxceleb-resnet34-LM"
    ]
    
    missing_models = []
    for model_path in required_models:
        if not os.path.exists(model_path):
            missing_models.append(model_path)
    
    if missing_models:
        error_msg = "The following required models are missing:\n"
        for model in missing_models:
            error_msg += f"  - {model}\n"
        error_msg += "\nPlease run 'python setup.py' to download the required models."
        return False, error_msg
    
    return True, "All models found."

def get_pipeline():
    """Load and cache the speaker diarization pipeline."""
    global pipeline, device
    if pipeline is None:
        # Check if models are downloaded
        models_ok, msg = check_models_exist()
        if not models_ok:
            raise Exception(msg)
        
        print("Loading pipeline from local cache...")
        # Load from the HuggingFace model ID, but with cache_dir pointing to our local data
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=hf_token,
            cache_dir="data",
        )
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        pipeline.to(device)
        print(f"✓ Pipeline loaded successfully on {device}!")
    return pipeline

def load_audio_manually(audio_file):
    """Load audio file manually using soundfile/pydub as fallback"""
    try:
        # Load audio with soundfile
        audio_data, sample_rate = sf.read(audio_file)
        # Convert to mono if stereo
        if len(audio_data.shape) > 1:
            audio_data = np.mean(audio_data, axis=1)
        # Convert to torch tensor
        audio_data = torch.from_numpy(audio_data).float()
        return {"waveform": audio_data.unsqueeze(0), "sample_rate": sample_rate}
    except ImportError:
        try:
            # Load with pydub
            audio = AudioSegment.from_file(audio_file)
            # Convert to mono and get raw data
            audio = audio.set_channels(1)
            audio_data = np.array(audio.get_array_of_samples(), dtype=np.float32) / 32768.0
            # Convert to torch tensor
            audio_data = torch.from_numpy(audio_data).float()
            sample_rate = audio.frame_rate
            return {"waveform": audio_data.unsqueeze(0), "sample_rate": sample_rate}
        except ImportError:
            raise Exception("Neither soundfile nor pydub is available for audio loading")

def diarize_audio(audio_file):
    """Perform speaker diarization on the uploaded audio file."""
    if audio_file is None:
        return "Please upload an audio file."

    try:
        # Load pipeline only when needed
        pipeline = get_pipeline()

        # Try to load audio manually first to avoid torchcodec issues
        try:
            audio_dict = load_audio_manually(audio_file)
            # Move to device
            audio_dict["waveform"] = audio_dict["waveform"].to(device)
            # Run diarization with manually loaded audio
            diarization = pipeline(audio_dict)
        except Exception as audio_load_error:
            # Fallback to letting pipeline handle it
            print(f"Manual audio loading failed: {audio_load_error}, trying pipeline default...")
            diarization = pipeline(audio_file)

        # Handle DiarizeOutput or Annotation
        if hasattr(diarization, 'speaker_diarization'):
            annotation = diarization.speaker_diarization
        else:
            annotation = diarization

        # Format the results
        results = []
        for turn, _, speaker in annotation.itertracks(yield_label=True):
            start_time = f"{turn.start:.1f}"
            end_time = f"{turn.end:.1f}"
            results.append(f"{speaker}: {start_time}s - {end_time}s")

        return "\n".join(results) if results else "No speakers detected in the audio."
    
    except Exception as e:
        return f"Error during diarization: {str(e)}"

# Create Gradio interface
iface = gr.Interface(
    fn=diarize_audio,
    inputs=gr.Audio(type="filepath", label="Upload Audio File (MP3 or WAV)"),
    outputs=gr.Textbox(label="Speaker Timestamps", lines=15),
    title="Speaker Diarization App",
    description="Upload an audio file to get timestamps of when each speaker is speaking."
)

if __name__ == "__main__":
    # Check models before starting the app
    models_ok, msg = check_models_exist()
    if not models_ok:
        print("\n" + "=" * 60)
        print("ERROR: Required models not found!")
        print("=" * 60)
        print(msg)
        print("=" * 60)
        print("\nTo download models, run:")
        print("  python setup.py")
        print("=" * 60 + "\n")
        sys.exit(1)
    
    print("\n" + "=" * 60)
    print("✓ All required models are available")
    print("=" * 60)
    print("Loading models on startup...")
    
    # Pre-load pipeline on startup
    try:
        get_pipeline()
        print("✓ Models loaded successfully!")
    except Exception as e:
        print(f"✗ Error loading models: {e}")
        sys.exit(1)
    
    print("=" * 60)
    print("Launching Gradio interface...\n")
    
    iface.launch()