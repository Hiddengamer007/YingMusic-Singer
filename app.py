import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

import gradio as gr
import torch

from singer.model import SAMPLE_RATE_48K, YingSinger

# =============================================================================
# Model Initialization
# =============================================================================

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading model on {device}...")

try:
    singer = YingSinger(singer_path="ckpts", device=device)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    singer = None


# =============================================================================
# Inference Function
# =============================================================================


def run_inference(
    timbre_audio: str,
    timbre_content: str,
    melody_audio: str,
    midi_file: str,
    lyrics: str,
    pitch_shift: int,
    cfg_strength: float,
    nfe_steps: int,
    sde_strength: float,
    seed: int,
) -> tuple[int, torch.Tensor]:
    """Run singing voice synthesis inference.

    Args:
        timbre_audio: Path to timbre reference audio.
        timbre_content: Text content of timbre reference.
        melody_audio: Path to melody reference audio.
        midi_file: Path to MIDI file (optional).
        lyrics: Target lyrics to synthesize.
        pitch_shift: Semitones to shift the melody key.
        cfg_strength: Classifier-free guidance strength.
        nfe_steps: Number of diffusion steps.
        sde_strength: Strength of SDE noise injection.
        seed: Random seed.

    Returns:
        Tuple of (sample_rate, audio_data).

    Raises:
        gr.Error: If model not loaded or required inputs missing.
    """
    if singer is None:
        raise gr.Error("Model not loaded. Please check the logs.")

    if not timbre_audio:
        raise gr.Error("Please upload the timbre reference audio.")
    if not timbre_content:
        raise gr.Error("Please enter the reference audio text content.")
    if not melody_audio and not midi_file:
        raise gr.Error("Please upload the melody reference audio or a MIDI file.")

    if midi_file:
        print(f"Using MIDI file for melody input: {midi_file}")
        gr.Info("Using MIDI file as melody input.")
        melody_audio = None
    else:
        print(f"Using audio file for melody input: {melody_audio}")
        gr.Info("Using audio file to extract the melody.")

    if not lyrics:
        raise gr.Error("Please enter the lyrics.")

    try:
        print(f"Starting inference with seed: {seed}")
        gen_wav = singer.inference(
            timbre_audio_path=timbre_audio,
            timbre_audio_content=timbre_content,
            melody_audio_path=melody_audio,
            midi_file=midi_file,
            lyrics=lyrics,
            pitch_shift=int(pitch_shift),
            cfg_strength=float(cfg_strength),
            nfe_steps=int(nfe_steps),
            sde_strength=float(sde_strength),
            seed=int(seed) if seed is not None else 2025,
        )
        return (SAMPLE_RATE_48K, gen_wav.numpy().T)
    except Exception as e:
        raise gr.Error(f"Generation failed: {str(e)}")


demo_inputs = [
    {
        "timbre_audio": "resources/audios/female.wav",
        "timbre_content": "冰刀划的圈，圈起了谁改变。",
        "melody_audio": "resources/audios/female__Rnb_Funk__下等马_clip_001.wav",
        "midi_file": None,
        "lyrics": "头抬起来，你表情别太奇怪，无大碍。没伤到脑袋，如果我下手太重，私密马赛。习武十载，没下山没谈恋爱，吃光后山七八亩菜，练就这套拳脚，莫以貌取人哉。暮色压台，擂鼓未衰，下一个谁还要来？速来领拜，别耽误我热蒸屉揭盖。",
        "cfg_strength": 4.0,
        "nfe_steps": 32,
        "seed": 666,
        "sde_strength": 0.3,
        "pitch_shift": 3,
    },
    {
        "timbre_audio": "resources/audios/male.wav",
        "timbre_content": "在爱的回归线，又期待相见。",
        "melody_audio": None,
        "midi_file": "resources/audios/female__Rnb_Funk__下等马_clip_001.mid",
        "lyrics": "头抬起来，你表情别太奇怪，无大碍。没伤到脑袋，如果我下手太重，私密马赛。习武十载，没下山没谈恋爱，吃光后山七八亩菜，练就这套拳脚，莫以貌取人哉。暮色压台，擂鼓未衰，下一个谁还要来？速来领拜，别耽误我热蒸屉揭盖。",
        "cfg_strength": 4.0,
        "nfe_steps": 64,
        "seed": 666,
        "sde_strength": 0,
        "pitch_shift": -9,
    },
    {
        "timbre_audio": "resources/audios/female_speech.wav",
        "timbre_content": "路人穿街过河，好景只有片刻，森林都会凋落，风吹走云朵。",
        "melody_audio": None,
        "midi_file": "resources/audios/female__Rnb_Funk__下等马_clip_001.mid",
        "lyrics": "心敞开来，你泪水别太奇怪，会好的。没伤到未来，如果我爱得太深，请原谅我。疗伤十载，没出门没再恋爱，吃光回忆七八亩菜，练就这心坚强，莫以泪洗面哉。曙光压台，希望未衰，新生活谁还要来？速来领爱，别耽误我心扉敞开怀。",
        "cfg_strength": 4.0,
        "nfe_steps": 64,
        "seed": 2025,
        "sde_strength": 0,
        "pitch_shift": -6,
    },
]


with gr.Blocks(title="YingSinger WebUI") as app:
    gr.Markdown(
        """
        <div style="text-align: center;">
            <h1>YingMusic-Singer Zero-Shot Singing Voice Synthesis & Editing</h1>
            <p style="font-size: 15px; color: #666;">
                Clone any singing voice and sing your own lyrics to any melody — no training required.
            </p>
        </div>
        """
    )

    gr.Markdown(
        """
        **How it works:** Upload a short clip of a voice you want to imitate (timbre), provide the
        melody you want it to sing (either an audio clip or a MIDI file), and type the lyrics.
        The model then generates a new singing performance in that voice, following the melody and your words.
        """
    )

    gr.Markdown(
        "### 1. Input Settings"
        "<br><small style='color:#888;'>Tip: Use dry/clean audio (no reverb, music, or background noise) "
        "for best results. The timbre and melody clips should ideally be the same language as your lyrics.</small>"
    )
    with gr.Row():
        with gr.Column():
            timbre_audio = gr.Audio(label="Timbre Reference Audio (Dry)", type="filepath")
            gr.Markdown(
                "<small style='color:#888;'>The voice to imitate. A 5–15s clean singing or speech clip works best.</small>"
            )
            timbre_content = gr.Textbox(
                label="Reference Audio Text Content",
                placeholder="Enter the exact spoken/sung text content of the reference audio above",
                lines=2,
            )
            gr.Markdown(
                "<small style='color:#888;'>This helps the model understand the voice. Must match what is actually said/sung in the clip.</small>"
            )

        with gr.Column():
            with gr.Tabs():
                with gr.Tab("Extract melody from audio"):
                    melody_audio = gr.Audio(label="Melody Reference Audio (Dry)", type="filepath")
                    gr.Markdown(
                        "<small style='color:#888;'>The tune to sing. A clean instrumental or acapella clip whose melody will be followed.</small>"
                    )
                with gr.Tab("Use MIDI file"):
                    midi_file = gr.File(label="MIDI File", file_types=[".mid", ".midi"])
                    gr.Markdown(
                        "<small style='color:#888;'>Alternatively, provide a MIDI file defining the exact notes/pitch of the melody.</small>"
                    )

            lyrics = gr.Textbox(label="Target Lyrics", placeholder="Enter the lyrics you want to synthesize", lines=2)
            gr.Markdown(
                "<small style='color:#888;'>The words the generated voice will sing, following the melody above. Punctuation adds natural pauses.</small>"
            )

    with gr.Accordion("Advanced Parameter Settings", open=True):
        gr.Markdown(
            "<small style='color:#888;'>These control the sound and quality. The defaults work well — tweak them if you want to experiment.</small>"
        )
        with gr.Row():
            pitch_shift = gr.Slider(
                minimum=-12, maximum=12, value=0, step=1,
                label="Pitch Shift (Semitones)",
                info="Shift the melody up (+) or down (-) to better fit the voice's natural range.",
            )
            cfg_strength = gr.Slider(
                minimum=1.0, maximum=10.0, value=4.0, step=0.1,
                label="CFG Strength (Guidance)",
                info="Higher = stricter adherence to the melody/lyrics but less natural; lower = more expressive but may drift.",
            )
            nfe_steps = gr.Slider(
                minimum=10, maximum=200, value=32, step=1,
                label="NFE Steps (Inference Steps)",
                info="More steps = higher quality but slower generation. 32 is fast, 64+ is higher quality.",
            )
            sde_strength = gr.Slider(
                minimum=0.0, maximum=1.0, value=0.3, step=0.01,
                label="SDE Strength",
                info="Adds variation/noise for a more natural, less robotic result. 0 = deterministic.",
            )
            seed = gr.Number(
                value=666, label="Random Seed", precision=0,
                info="Same seed + same inputs = same output. Change it to get a different variation.",
            )

    submit_btn = gr.Button("🎤 Generate", variant="primary")

    gr.Markdown(
        "### 2. Generation Results"
        "<br><small style='color:#888;'>Not happy with the result? Try a different <b>Seed</b> or <b>SDE Strength</b>, "
        "or check that your reference audio is clean. Higher <b>NFE Steps</b> generally improves quality.</small>"
    )
    output_audio = gr.Audio(label="Synthesized Audio", type="numpy", interactive=False)

    gr.Examples(
        examples=[
            [
                x["timbre_audio"],
                x["timbre_content"],
                x["melody_audio"],
                x["midi_file"],
                x["lyrics"],
                x["pitch_shift"],
                x["cfg_strength"],
                x["nfe_steps"],
                x["sde_strength"],
                x["seed"],
            ]
            for x in demo_inputs
        ],
        inputs=[
            timbre_audio,
            timbre_content,
            melody_audio,
            midi_file,
            lyrics,
            pitch_shift,
            cfg_strength,
            nfe_steps,
            sde_strength,
            seed,
        ],
        label="Example Inputs (Click any row to load it)",
        examples_per_page=10,
    )

    submit_btn.click(
        fn=run_inference,
        inputs=[
            timbre_audio,
            timbre_content,
            melody_audio,
            midi_file,
            lyrics,
            pitch_shift,
            cfg_strength,
            nfe_steps,
            sde_strength,
            seed,
        ],
        outputs=output_audio,
    )

if __name__ == "__main__":
    app.launch(allowed_paths=["."])
