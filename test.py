from tts.engine import generate
import sounddevice as sd

SAMPLE_RATE = 48000


def play(pcm):
    if pcm is None:
        print("No audio generated.")
        return

    sd.play(pcm, SAMPLE_RATE)
    sd.wait()


print("TTS Test mode. Type text and press Enter.\n")

while True:
    text = input("> ")

    if text.lower() in ["exit", "quit"]:
        break

    audio = generate(text, pitch=1.25)

    print("Generated:", "OK" if audio is not None else "NONE")
    play(audio)
