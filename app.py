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

        # Format the results - only show new lines when speaker changes
        results = []
        current_speaker = None
        start_time = None
        end_time = None

        for turn, _, speaker in annotation.itertracks(yield_label=True):
            if speaker != current_speaker:
                # Speaker changed - save previous speaker's time range
                if current_speaker is not None:
                    results.append(f"{current_speaker}: {start_time}s - {end_time}s")
                # Start tracking new speaker
                current_speaker = speaker
                start_time = f"{turn.start:.1f}"
            # Always update end time for current speaker
            end_time = f"{turn.end:.1f}"

        # Add the last speaker's time range
        if current_speaker is not None:
            results.append(f"{current_speaker}: {start_time}s - {end_time}s")

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
        pipeline = get_pipeline()
        print("✓ Pipeline loaded into memory!")

        # Warmup inference to force models into GPU VRAM
        print("Warming up models (loading into GPU VRAM)...")
        print("  Creating temporary audio file...")

        # Create a real temporary WAV file (same approach as diarize_audio uses)
        import tempfile
        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".wav")
        try:
            # Create 30 seconds of synthetic speech-like audio to trigger all models
            sample_rate = 16000
            duration = 30  # 30 seconds to ensure full processing
            t = np.linspace(0, duration, duration * sample_rate, dtype=np.float32)

            # Create speech-like signal with multiple "speakers"
            # Speaker 1: 0-15 seconds (fundamental freq ~150 Hz, male-like)
            speaker1 = np.sin(2 * np.pi * 150 * t[:15*sample_rate])
            speaker1 += np.sin(2 * np.pi * 300 * t[:15*sample_rate]) * 0.5  # Harmonics
            speaker1 += np.sin(2 * np.pi * 450 * t[:15*sample_rate]) * 0.3
            speaker1 *= (0.3 + 0.7 * np.abs(np.sin(2 * np.pi * 3 * t[:15*sample_rate])))  # Amplitude modulation

            # Speaker 2: 15-30 seconds (fundamental freq ~220 Hz, female-like)
            speaker2 = np.sin(2 * np.pi * 220 * t[15*sample_rate:])
            speaker2 += np.sin(2 * np.pi * 440 * t[15*sample_rate:]) * 0.5
            speaker2 += np.sin(2 * np.pi * 660 * t[15*sample_rate:]) * 0.3
            speaker2 *= (0.3 + 0.7 * np.abs(np.sin(2 * np.pi * 4 * t[15*sample_rate:])))

            # Combine speakers and add slight noise
            dummy_audio = np.concatenate([speaker1, speaker2])
            dummy_audio = dummy_audio * 0.3 + np.random.randn(len(dummy_audio)).astype(np.float32) * 0.01
            dummy_audio = dummy_audio.astype(np.float32)

            sf.write(tmp_path, dummy_audio, sample_rate)

            print(f"  Running warmup inference on temporary file...")
            print(f"  (This will load all neural network models into GPU VRAM)")

            # Call pipeline exactly like diarize_audio does
            diarization_result = pipeline(tmp_path)

            print("✓ Warmup completed!")
            print("✓ All models are now loaded in GPU VRAM!")

            # Show if we got results
            if hasattr(diarization_result, 'speaker_diarization'):
                annotation = diarization_result.speaker_diarization
            else:
                annotation = diarization_result

            num_speakers = len(list(annotation.labels()))
            print(f"  Warmup detected {num_speakers} speakers in random noise (expected)")

        except Exception as warmup_error:
            print(f"✗ Error during warmup inference: {warmup_error}")
            import traceback
            traceback.print_exc()
            print("\n⚠ Warning: Warmup failed, models will load on first actual use")
        finally:
            # Clean up
            os.close(tmp_fd)
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    except Exception as e:
        print(f"✗ Error loading pipeline: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print("=" * 60)
    print("Launching Gradio interface...\n")

    iface.launch()