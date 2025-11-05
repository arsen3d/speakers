#when launching this app, use uv command
import gradio as gr
import os
from dotenv import load_dotenv
import numpy as np
import torch

# Load environment variables from .env file
load_dotenv()

# Login to Hugging Face if token is available
hf_token = os.getenv('HF_TOKEN')
if hf_token:
    try:
        from huggingface_hub import login
        login(hf_token)
    except Exception as e:
        print(f"Failed to login to Hugging Face: {e}")
        print("Continuing without login...")

# Lazy-loaded pipeline
pipeline = None

def get_pipeline():
    global pipeline
    if pipeline is None:
        try:
            from pyannote.audio import Pipeline
            pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")
            print("Pipeline loaded successfully!")
        except Exception as e:
            error_msg = str(e)
            if "403" in error_msg or "gated" in error_msg.lower() or "forbidden" in error_msg.lower():
                raise Exception("The pyannote/speaker-diarization-3.1 model is gated and requires special access. Please visit https://huggingface.co/pyannote/speaker-diarization-3.1 to request access.")
            else:
                raise Exception(f"Failed to load pipeline: {error_msg}")
    return pipeline

def load_audio_manually(audio_file):
    """Load audio file manually using soundfile/pydub as fallback"""
    try:
        import soundfile as sf
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
            from pydub import AudioSegment
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
    if audio_file is None:
        return "Please upload an audio file."

    try:
        # Load pipeline only when needed
        pipeline = get_pipeline()

        # Try to load audio manually first to avoid torchcodec issues
        try:
            audio_dict = load_audio_manually(audio_file)
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

        return "\n".join(results)
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
    iface.launch()