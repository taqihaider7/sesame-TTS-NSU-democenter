import gradio as gr
import numpy as np

from generator import load_csm_1b, generate_streaming_audio

# Load the TTS model once at startup
print("Loading TTS model...")
generator = load_csm_1b("eustlb/csm-1b")
print("Model loaded. Ready for inference.")

    
# Define the inference function for Gradio
def text_to_speech(text, chunk_token_size: int = 20):
    """
    Generate speech from input text using the CSM-1B generator.

    Args:
        text (str): Text prompt to synthesize.
        chunk_token_size (int): Number of tokens per streaming chunk.

    Returns:
        Tuple[int, np.ndarray]: Sample rate and generated audio waveform.
    """

    if not text:
        return None, None

    text = [
        {"role": "0", "content": [{"type": "text", "text": 'how are you'}
    ]
    
    # Generate audio without playing locally
    audio_array = generate_streaming_audio(
        generator,
        conversation=text,
        play_audio=False,
        chunk_token_size=chunk_token_size
    )

    # CSM-1B outputs at 24 kHz
    sample_rate = 24000

    # Gradio expects a tuple of (sample_rate, numpy array)
    return sample_rate, audio_array

# Create Gradio interface
iface = gr.Interface(
    fn=text_to_speech,
    inputs=[
        gr.Textbox(lines=3, label="Input Text", placeholder="Enter text to synthesize..."),
        gr.Slider(5, 50, value=20, step=5, label="Chunk Token Size")
    ],
    outputs=gr.Audio(label="Generated Speech", type="numpy"),
    title="Text-to-Speech with CSM-1B",
    description="Enter text and click 'Submit' to generate speech using the CSM-1B model.",
    flagging_mode="never"
)

if __name__ == "__main__":
    iface.launch(server_name="0.0.0.0", server_port=1215, share=False)
