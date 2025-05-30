import gradio as gr
import numpy as np
import torch
import torchaudio
from generator import Segment, load_csm_1b

# Load the TTS model once at startup
print("Loading TTS model...")
generator = load_csm_1b("eustlb/csm-1b")
print("Model loaded. Ready for inference.")


def prepare_prompt(text: str, speaker: int, audio_path: str) -> Segment:
    audio_tensor, _ = load_prompt_audio(audio_path)
    return Segment(text=text, speaker=speaker, audio=audio_tensor)

def load_prompt_audio(audio_path: str) -> torch.Tensor:
    audio_tensor, sample_rate = torchaudio.load(audio_path)
    if audio_tensor.shape[0] != 1:
        gr.Warning("Warning: Audio prompt is multi-channel, converting to mono.", duration=15)
        audio_tensor = audio_tensor.mean(dim=0)
    audio_tensor = audio_tensor.squeeze(0)
    if sample_rate != generator.sample_rate:
        audio_tensor = torchaudio.functional.resample(
            audio_tensor, orig_freq=sample_rate, new_freq=generator.sample_rate
        )
    return audio_tensor, generator.sample_rate


def infer(
    gen_conversation_input
) -> tuple[np.ndarray, int]:

    try:
        return _infer(
            gen_conversation_input
        )
    except ValueError as e:
        raise gr.Error(f"Error generating audio: {e}", duration=120)


def _infer(
    gen_conversation_input
) -> tuple[np.ndarray, int]:
    SPEAKER_PROMPTS = {
        "conversational_a": {
            "text": (
                "like revising for an exam I'd have to try and like keep up the momentum because I'd "
                "start really early I'd be like okay I'm gonna start revising now and then like "
                "you're revising for ages and then I just like start losing steam I didn't do that "
                "for the exam we had recently to be fair that was a more of a last minute scenario "
                "but like yeah I'm trying to like yeah I noticed this yesterday that like Mondays I "
                "sort of start the day with this not like a panic but like a"
            ),
            "audio": "/work/Train-Practice/CSM-transformer/csm-streaming-tf/csm-1b/prompts/conversational_a.wav",
        },
        "conversational_b": {
            "text": (
                "like a super Mario level. Like it's very like high detail. And like, once you get "
                "into the park, it just like, everything looks like a computer game and they have all "
                "these, like, you know, if, if there's like a, you know, like in a Mario game, they "
                "will have like a question block. And if you like, you know, punch it, a coin will "
                "come out. So like everyone, when they come into the park, they get like this little "
                "bracelet and then you can go punching question blocks around."
            ),
            "audio": "/work/Train-Practice/CSM-transformer/csm-streaming-tf/csm-1b/prompts/conversational_b.wav",},
    }

    
    text_prompt_speaker_a = SPEAKER_PROMPTS["conversational_a"]["text"]
    audio_prompt_speaker_a = SPEAKER_PROMPTS["conversational_a"]["audio"]
    text_prompt_speaker_b = SPEAKER_PROMPTS["conversational_b"]["text"]
    audio_prompt_speaker_b = SPEAKER_PROMPTS["conversational_b"]["audio"]
    
    audio_prompt_a = prepare_prompt(text_prompt_speaker_a, 0, audio_prompt_speaker_a)
    audio_prompt_b = prepare_prompt(text_prompt_speaker_b, 1, audio_prompt_speaker_b)

    prompt_segments: list[Segment] = [audio_prompt_a, audio_prompt_b]
    generated_segments: list[Segment] = []

    conversation_lines = [line.strip() for line in gen_conversation_input.strip().split("\n") if line.strip()]
    for i, line in enumerate(conversation_lines):
        # Alternating speakers A and B, starting with A
        speaker_id = i % 2

        audio_tensor = generator.generate(
            text=line,
            speaker=speaker_id,
            context=prompt_segments + generated_segments,
            max_audio_length_ms=30_000,
        )
        generated_segments.append(Segment(text=line, speaker=speaker_id, audio=audio_tensor))

    audio_tensors = [segment.audio for segment in generated_segments]
    audio_tensor = torch.cat(audio_tensors, dim=0)


    audio_tensor = torchaudio.functional.resample(
        audio_tensor, orig_freq=24000, new_freq=generator.sample_rate
    )

    audio_array = (audio_tensor * 32768).to(torch.int16).cpu().numpy()

    return generator.sample_rate, audio_array   


# Create Gradio interface
iface = gr.Interface(
    fn=infer,
    inputs=[
        gr.Textbox(lines=3, label="Input Text", placeholder="Enter text to synthesize..."),
        #gr.Slider(5, 50, value=20, step=5, label="Chunk Token Size")
    ],
    outputs=gr.Audio(label="Generated Speech", type="numpy"),
    title="Text-to-Speech with CSM-1B",
    description="Enter text and click 'Submit' to generate speech using the CSM-1B model.",
    flagging_mode="never"
)

if __name__ == "__main__":
    iface.launch(server_name="0.0.0.0", server_port=1215, share=False)
