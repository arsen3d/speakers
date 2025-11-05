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
from faster_whisper import WhisperModel

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
whisper_model = None
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

def get_whisper_model():
    """Load and cache the Whisper model."""
    global whisper_model, device
    if whisper_model is None:
        print("Loading Whisper model...")
        device_type = "cuda" if torch.cuda.is_available() else "cpu"
        compute_type = "float16" if device_type == "cuda" else "int8"

        whisper_model = WhisperModel(
            "tiny",
            device=device_type,
            compute_type=compute_type,
            download_root="data/whisper"
        )
        print(f"✓ Whisper model loaded successfully on {device_type}!")
    return whisper_model

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

def transcribe_and_diarize(audio_file):
    """Perform transcription and speaker diarization, then combine them."""
    if audio_file is None:
        return "Please upload an audio file."

    try:
        # Load both models
        pipeline = get_pipeline()
        whisper = get_whisper_model()

        # Step 1: Transcribe with Whisper (word-level timestamps)
        print("Transcribing audio...")
        segments, info = whisper.transcribe(
            audio_file,
            word_timestamps=True,
            language="en"  # Set to None for auto-detection, or specify language
        )

        # Collect all words with timestamps
        words = []
        for segment in segments:
            if hasattr(segment, 'words') and segment.words:
                for word in segment.words:
                    words.append({
                        'start': word.start,
                        'end': word.end,
                        'text': word.word
                    })

        # Step 2: Perform speaker diarization
        print("Performing speaker diarization...")
        try:
            audio_dict = load_audio_manually(audio_file)
            audio_dict["waveform"] = audio_dict["waveform"].to(device)
            diarization = pipeline(audio_dict)
        except Exception as audio_load_error:
            print(f"Manual audio loading failed: {audio_load_error}, trying pipeline default...")
            diarization = pipeline(audio_file)

        # Handle DiarizeOutput or Annotation
        if hasattr(diarization, 'speaker_diarization'):
            annotation = diarization.speaker_diarization
        else:
            annotation = diarization

        # Step 3: Combine transcription with speaker labels
        print("Combining transcription with speaker labels...")

        # Build speaker segments list
        speaker_segments = []
        for turn, _, speaker in annotation.itertracks(yield_label=True):
            speaker_segments.append({
                'start': turn.start,
                'end': turn.end,
                'speaker': speaker
            })

        # Debug: Print speaker segments
        print(f"DEBUG: Found {len(speaker_segments)} speaker segments:")
        for seg in speaker_segments:
            print(f"  {seg['speaker']}: {seg['start']:.2f}s - {seg['end']:.2f}s")

        # Assign speakers to words
        unassigned_count = 0
        for word in words:
            word_mid = (word['start'] + word['end']) / 2
            # Find which speaker segment this word belongs to
            assigned_speaker = None
            min_distance = float('inf')

            for segment in speaker_segments:
                if segment['start'] <= word_mid <= segment['end']:
                    # Word is inside this segment - perfect match
                    assigned_speaker = segment['speaker']
                    break
                else:
                    # Calculate distance to this segment
                    if word_mid < segment['start']:
                        distance = segment['start'] - word_mid
                    else:  # word_mid > segment['end']
                        distance = word_mid - segment['end']

                    # Keep track of nearest segment
                    if distance < min_distance:
                        min_distance = distance
                        assigned_speaker = segment['speaker']

            word['speaker'] = assigned_speaker if assigned_speaker else "UNKNOWN"
            if assigned_speaker is None:
                unassigned_count += 1

        # Debug: Print speaker assignment summary
        speaker_counts = {}
        for word in words:
            speaker = word['speaker']
            speaker_counts[speaker] = speaker_counts.get(speaker, 0) + 1

        print(f"DEBUG: Word speaker assignment summary:")
        for speaker, count in speaker_counts.items():
            print(f"  {speaker}: {count} words")
        if unassigned_count > 0:
            print(f"  WARNING: {unassigned_count} words could not be assigned to any speaker!")

        # Step 4: Format output - group consecutive words by same speaker
        results = []
        current_speaker = None
        current_text = []
        current_start = None

        for word in words:
            if word['speaker'] != current_speaker:
                # Speaker changed - save previous speaker's text
                if current_speaker is not None and current_text:
                    text = ''.join(current_text).strip()
                    results.append(f"{current_speaker} [{current_start:.1f}s]: {text}")

                # Start new speaker segment
                current_speaker = word['speaker']
                current_text = [word['text']]
                current_start = word['start']
            else:
                # Same speaker continues
                current_text.append(word['text'])

        # Add the last speaker's text
        if current_speaker is not None and current_text:
            text = ''.join(current_text).strip()
            results.append(f"{current_speaker} [{current_start:.1f}s]: {text}")

        return "\n\n".join(results) if results else "No speakers or transcription detected in the audio."

    except Exception as e:
        import traceback
        error_msg = f"Error during transcription/diarization: {str(e)}\n\n"
        error_msg += traceback.format_exc()
        return error_msg

# Create Gradio interface
iface = gr.Interface(
    fn=transcribe_and_diarize,
    inputs=gr.Audio(type="filepath", label="Upload Audio File (MP3 or WAV)"),
    outputs=gr.Textbox(label="Transcription with Speaker Labels", lines=20),
    title="Speaker Diarization + Transcription App",
    description="Upload an audio file to get a transcription with speaker labels and timestamps."
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

    # Pre-load pipeline and Whisper on startup
    try:
        pipeline = get_pipeline()
        print("✓ Pipeline loaded into memory!")

        whisper = get_whisper_model()
        print("✓ Whisper model loaded into memory!")

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

            # Call pipeline exactly like transcribe_and_diarize does
            diarization_result = pipeline(tmp_path)

            # Warmup Whisper too
            print("  Warming up Whisper model...")
            whisper_segments, _ = whisper.transcribe(tmp_path, word_timestamps=True)
            # Consume the generator
            list(whisper_segments)

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