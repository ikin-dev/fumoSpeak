import threading
import time
from queue import Queue
import tkinter as tk
from tkinter import ttk
from tkinter.scrolledtext import ScrolledText
import importlib.util
import shutil
import json
import os
import torch
import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel
from faster_whisper.vad import VadOptions, get_speech_timestamps
from tts.engine import generate

# settings
SETTINGS_FILE = "settings.json"


def load_settings():
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_settings(data):
    try:
        with open(SETTINGS_FILE, "w") as f:
            json.dump(data, f, indent=2)
    except Exception:
        pass


def is_installed(pkg):
    return importlib.util.find_spec(pkg) is not None


def cuda_info():
    return {
        "available": torch.cuda.is_available(),
        "name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


MODEL_MAP = {
    "tiny.en (~75 MB)": "tiny.en",
    "base.en (~145 MB)": "base.en",
    "small.en (~488 MB)": "small.en",
    "medium.en (~1.5 GB)": "medium.en",
    "distil-small.en (~500 MB)": "distil-small.en",
    "distil-medium.en (~1.5 GB)": "distil-medium.en",
}

# audio capture config
RATE = 16000
BLOCKSIZE = 4000

# speech segmentation (VAD) config
MIN_SPEECH_DURATION = 1.0   # seconds - shortest utterance worth sending to Whisper
MAX_SPEECH_DURATION = 8.0   # seconds - force a cut even if the person is still talking
DROP_START_SILENCE = 0.25   # seconds - padding kept around detected speech
PAUSE_DURATION = 1.0        # seconds of silence that marks "utterance is finished"
VAD_THRESHOLD = 0.5         # Silero speech-probability threshold
VAD_POLL_INTERVAL = 0.1     # seconds between buffer checks in the worker loop

class VoiceApp:
    def __init__(self, root):
        self.root = root
        self.root.title("fumoSpeak")
        self.root.geometry("500x500")
        self.root.resizable(False, True)

        try: # this is still weird
            self.root.iconphoto(False, tk.PhotoImage(file="assets/icon.png"))
            self.root.iconbitmap("assets/icon.ico")
        except Exception as e:
            print(f"Icon load skipped: {e}")

        self.running = False
        self.model = None
        self.stream = None
        self.audio_queue = Queue()
        self.buffer = []
        self.buffer_lock = threading.Lock()
        self.stop_audio_event = threading.Event()
        self.settings = load_settings()

        self.build_ui()
        self.load_saved_settings()
        threading.Thread(target=self.playback_worker, daemon=True).start()
        self.startup_check()

    # settings
    def load_saved_settings(self):
        pitch = self.settings.get("pitch", 1.25)

        self.model_var.set(self.settings.get("model", "medium.en (~1.5 GB)"))
        self.device_var.set(self.settings.get("device", "CPU"))
        self.pitch_var.set(pitch)
        self.pitch_label.config(text=f"{pitch:.2f}x")

    def persist_settings(self):
        save_settings(
            {
                "model": self.model_var.get(),
                "device": self.device_var.get(),
                "input": self.input_var.get(),
                "output": self.output_var.get(),
                "pitch": float(self.pitch_var.get()),
            }
        )

    def on_setting_change(self, reason="Setting"):
        if self.running:
            self.log(f"{reason} changed → stopping session")
            self.stop()
        self.stop_audio()
        self.persist_settings()

    def set_running_state(self, running: bool):
        self.running = running
        self.start_btn.config(state="disabled" if running else "normal")
        self.stop_btn.config(state="normal" if running else "disabled")

    def stop_audio(self):
        self.stop_audio_event.set()
        sd.stop()

    def log(self, text):
        self.root.after(0, lambda: self._log(text))

    def _log(self, text):
        self.log_box.insert(tk.END, text + "\n")
        self.log_box.see(tk.END)

    # check for depencencies, ffmpeg and cuda
    def startup_check(self):
        self.log("=== System Check ===")

        checks = {
            "numpy": is_installed("numpy"),
            "torch": is_installed("torch"),
            "sounddevice": is_installed("sounddevice"),
            "faster-whisper": is_installed("faster_whisper"),
        }

        for k, v in checks.items():
            self.log(f"{'✔' if v else '✖'} {k}")

        self.log(f"{'✔ ffmpeg' if shutil.which('ffmpeg') else '✖ ffmpeg: Not installed'}")
        cuda = cuda_info()
        self.log(f"{'✔ CUDA: ' + str(cuda['name']) if cuda['available'] else '✖ CUDA: Not available'}")
        self.log("====================")

    # ui
    def build_ui(self):
            top = ttk.Frame(self.root)
            top.pack(fill="x", padx=10, pady=10)
            
            for i in range(5):
                top.columnconfigure(i, weight=1)

            # model select
            ttk.Label(top, text="Model").grid(row=0, column=0, sticky="w", padx=(0, 6), pady=6)
            self.model_var = tk.StringVar()
            self.model_combo = ttk.Combobox(
                top, textvariable=self.model_var, values=list(MODEL_MAP.keys()), width=25
            )
            self.model_combo.grid(row=0, column=1, padx=6, pady=6, sticky="ew")

            # cpu and gpu select
            ttk.Label(top, text="Device").grid(row=0, column=2, sticky="w", padx=(10, 6), pady=6)
            self.device_var = tk.StringVar(value="CPU")
            self.device_combo = ttk.Combobox(
                top, textvariable=self.device_var, values=["CPU", "CUDA (GPU)"], width=15
            )
            self.device_combo.grid(row=0, column=3, padx=6, pady=6, sticky="ew")

            # input and output devices
            devices = sd.query_devices()

            input_devices = [
                f"{i}: {d['name']}"
                for i, d in enumerate(devices)
                if d["max_input_channels"] > 0
            ]

            output_devices = [
                f"{i}: {d['name']}"
                for i, d in enumerate(devices)
                if d["max_output_channels"] > 0
            ]

            self.input_var = tk.StringVar()
            self.output_var = tk.StringVar()

            ttk.Label(top, text="Input").grid(row=1, column=0, sticky="w", padx=(0, 6), pady=4)
            self.input_combo = ttk.Combobox(top, textvariable=self.input_var, values=input_devices)
            self.input_combo.grid(row=1, column=1, columnspan=4, sticky="ew", padx=6, pady=4)

            ttk.Label(top, text="Output").grid(row=2, column=0, sticky="w", padx=(0, 6), pady=4)
            self.output_combo = ttk.Combobox(top, textvariable=self.output_var, values=output_devices)
            self.output_combo.grid(row=2, column=1, columnspan=4, sticky="ew", padx=6, pady=4)

            if input_devices:
                self.input_combo.current(0)
            if output_devices:
                self.output_combo.current(0)

            # pitch slider
            self.pitch_var = tk.DoubleVar(value=1.25)
            ttk.Label(top, text="Pitch").grid(row=3, column=0, sticky="w", padx=(0, 6), pady=6)
            self.pitch_slider = ttk.Scale(
                top,
                from_=0.5,
                to=3.0,
                variable=self.pitch_var,
                orient="horizontal"
            )
            self.pitch_slider.grid(row=3, column=1, columnspan=3, sticky="ew", padx=6, pady=6)
            self.pitch_label = ttk.Label(top, text="1.25x")
            self.pitch_label.grid(row=3, column=4, padx=(6, 0))

            def update_pitch_label(*_):
                v = round(self.pitch_var.get() / 0.05) * 0.05
                v = max(0.5, min(3.0, v))
                self.pitch_var.set(v)
                self.pitch_label.config(text=f"{v:.2f}x")

            self.pitch_slider.config(command=update_pitch_label)

            # start and stop buttons
            btns = ttk.Frame(self.root)
            btns.pack(fill="x", padx=10, pady=5)

            self.start_btn = ttk.Button(btns, text="Start", command=self.start)
            self.stop_btn = ttk.Button(btns, text="Stop", command=self.stop, state="disabled")
            self.start_btn.pack(side="left", padx=6, pady=4)
            self.stop_btn.pack(side="left", padx=6, pady=4)

            # logs
            self.log_box = ScrolledText(self.root, height=20)
            self.log_box.pack(fill="both", expand=True, padx=10, pady=10)
        
    # audio
    def playback_worker(self):
        while True:
            audio = self.audio_queue.get()

            if audio is None:
                continue

            if self.stop_audio_event.is_set():
                continue

            try:
                output_index = int(self.output_var.get().split(":")[0])

                sd.stop()
                sd.play(audio, 48000, device=output_index)

                while sd.get_stream().active:
                    if self.stop_audio_event.is_set():
                        sd.stop()
                        break
                    sd.sleep(20)

            except Exception as e:
                self.log(f"Playback error: {e}")

    # tts
    def speak(self, text):
        try:
            pitch = float(self.pitch_var.get())
            audio = generate(text, pitch)
            if audio is not None:
                self.audio_queue.put(audio)

        except Exception as e:
            self.log(f"TTS error: {e}")

    # mic callback
    def callback(self, indata, frames, time_info, status):
        if status:
            self.log(str(status))

        with self.buffer_lock:
            self.buffer.append(indata.copy())

    # transcription
    def worker(self):
        # vad decides when the sentence is done and ready to send
        vad_options = VadOptions(
            threshold=VAD_THRESHOLD,
            min_speech_duration_ms=int(MIN_SPEECH_DURATION * 1000),
            max_speech_duration_s=MAX_SPEECH_DURATION,
            min_silence_duration_ms=int(PAUSE_DURATION * 1000),
            speech_pad_ms=int(DROP_START_SILENCE * 1000),
        )

        pending = np.zeros((0,), dtype=np.float32)

        while self.running:
            time.sleep(VAD_POLL_INTERVAL)

            with self.buffer_lock:
                if self.buffer:
                    new_audio = np.concatenate(self.buffer).flatten()
                    self.buffer.clear()
                else:
                    new_audio = None

            if new_audio is not None and new_audio.size:
                pending = np.concatenate([pending, new_audio])

            if pending.size == 0:
                continue

            duration = pending.size / RATE

            try:
                speech_segments = get_speech_timestamps(pending, vad_options, sampling_rate=RATE)
            except Exception as e:
                self.log(f"VAD error: {e}")
                continue

            if not speech_segments:
                # Nothing but silence so far - don't let this grow forever.
                if duration >= MAX_SPEECH_DURATION:
                    pending = np.zeros((0,), dtype=np.float32)
                continue

            last_seg = speech_segments[-1]
            trailing_silence = duration - (last_seg["end"] / RATE)
            utterance_done = trailing_silence >= PAUSE_DURATION
            hit_max_duration = duration >= MAX_SPEECH_DURATION

            if not (utterance_done or hit_max_duration):
                continue  # still mid-utterance, keep listening

            cut_sample = last_seg["end"] if utterance_done else int(MAX_SPEECH_DURATION * RATE)
            first_sample = min(speech_segments[0]["start"], cut_sample)

            utterance = pending[first_sample:cut_sample]
            pending = pending[cut_sample:]

            if utterance.size / RATE < MIN_SPEECH_DURATION:
                continue  # too short to be worth sending to Whisper

            try:
                segments, _ = self.model.transcribe(
                    utterance,
                    language="en",
                    beam_size=1,
                    best_of=1,
                    temperature=0.0,
                )

                text = " ".join(s.text.strip() for s in segments).strip()

                if text:
                    self.log("> " + text)
                    self.speak(text)

            except Exception as e:
                self.log(f"Transcription error: {e}")

    # start and stop session
    def start(self):
        if self.running:
            return

        try:
            self.stop_audio_event.clear()
            self.persist_settings()

            model_name = MODEL_MAP.get(self.model_var.get(), "tiny.en")
            device_ui = self.device_var.get()
            device = "cuda" if device_ui == "CUDA (GPU)" else "cpu"
            compute = "float16" if device == "cuda" else "int8"

            self.log(f"Loading model: {model_name}")

            self.model = WhisperModel(
                model_name,
                device=device,
                compute_type=compute,
            )

            input_index = int(self.input_var.get().split(":")[0])
            self.stream = sd.InputStream(
                samplerate=RATE,
                channels=1,
                dtype=np.float32,
                blocksize=BLOCKSIZE,
                device=input_index,
                callback=self.callback,
            )

            self.stream.start()
            self.set_running_state(True)
            threading.Thread(target=self.worker, daemon=True).start()
            self.log("Listening...")

        except Exception as e:
            self.log(f"Start failed: {e}")

    def stop(self):
        self.set_running_state(False)
        self.stop_audio()
        self.running = False

        try:
            if self.stream:
                self.stream.stop()
                self.stream.close()
                self.stream = None
        except Exception as e:
            self.log(f"Stop error: {e}")

        self.log("Stopped")


root = tk.Tk()
app = VoiceApp(root)
root.mainloop()