# converted from js

import os
import random
import numpy as np

from tts.data import (
    alphabet,
    alphabetJP,
    small_to_large,
    silent,
    exclaim,
    repeat,
)

SAMPLE_DIR = os.path.join(os.path.dirname(__file__), "PCM")
SAMPLE_RATE = 48000
CHANNELS = 1
BYTES_PER_SAMPLE = 2
SAMPLES_PER_MS = SAMPLE_RATE / 1000

sample_cache = {}


def read_pcm(filename):
    path = os.path.join(SAMPLE_DIR, filename)

    if path not in sample_cache:
        with open(path, "rb") as f:
            sample_cache[path] = np.frombuffer(
                f.read(),
                dtype=np.int16
            )

    return sample_cache[path]


def duration(filename):
    return len(read_pcm(filename))


def get_filename(char):
    if char in alphabet:
        return alphabet[char]

    if char in alphabetJP:
        return alphabetJP[char]

    return random.choice(list(alphabet.values()))


def normalize(message):
    out = ""

    for ch in message:
        utf = ord(ch)
        c = ch.lower()

        if 0x30A1 <= utf <= 0x30F6:
            c = chr(utf - 0x30A1 + 0x3041)

        if c in repeat:
            last = out[-1] if out else ""

            if last in "あかさたなはまやらわ":
                c = "あ"
            elif last in "いきしちにひみり":
                c = "い"
            elif last in "うくすつぬふむゆる":
                c = "う"
            elif last in "えけせてねへめれ":
                c = "え"
            elif last in "おこそとのほもよろを":
                c = "お"
            elif last == "ん":
                c = "ん"

        out += c

    return out


def tokenize(message):
    tokens = []

    while message:
        match = ""

        for token in alphabetJP.keys():
            if message.startswith(token) and len(token) > len(match):
                match = token

        if match:
            tokens.append(match)
            message = message[len(match):]
        else:
            tokens.append(message[0])
            message = message[1:]

    return tokens


def normalize_tokens(tokens):
    return [small_to_large.get(t, t) for t in tokens]


def spacing(token, is_exclaim):
    if token in alphabetJP or token == "　":
        return 100

    if token == "っ":
        return 160

    return 58


def message_duration(tokens, pitch, is_exclaim):
    total = int(duration(get_filename(tokens[0])) / pitch)

    for token in tokens[1:-1]:
        total += int(spacing(token, is_exclaim) * SAMPLES_PER_MS)

    total += int(duration(get_filename(tokens[-1])) / pitch)

    return total


def resample_nearest(data, pitch):
    """
    Replicates the original JS pitch shifting.
    """
    length = max(1, int(len(data) / pitch))

    idx = np.round(
        np.arange(length) * pitch
    ).astype(np.int32)

    idx = np.clip(idx, 0, len(data) - 1)

    return data[idx]


def generate(message, pitch=1.25):
    message = normalize_tokens(
        tokenize(
            normalize(message)
        )
    )

    if not any(t not in silent for t in message):
        return None

    is_exclaim = message[-1] in exclaim

    if is_exclaim:
        pitch += 0.04

    total_samples = message_duration(
        message,
        pitch,
        is_exclaim
    )

    output = np.zeros(
        total_samples,
        dtype=np.int32
    )

    pos_ms = 0

    for token in message:
        if token in silent:
            pos_ms += spacing(token, is_exclaim)
            continue

        filename = get_filename(token)
        data = read_pcm(filename)

        target_pitch = (
            pitch +
            random.randint(-5, 5) / 100
        )

        sample = resample_nearest(
            data,
            target_pitch
        )

        start = int(
            pos_ms *
            SAMPLES_PER_MS
        )

        end = min(
            start + len(sample),
            len(output)
        )

        output[start:end] += sample[: end - start]

        pos_ms += spacing(
            token,
            is_exclaim
        )

    output = np.clip(
        output,
        -32768,
        32767
    ).astype(np.int16)

    return output


def get_compressed(message, pitch=1.25):
    import subprocess

    pcm = generate(message, pitch)

    if pcm is None:
        return None

    proc = subprocess.Popen(
        [
            "ffmpeg",
            "-f",
            "s16le",
            "-ar",
            str(SAMPLE_RATE),
            "-ac",
            str(CHANNELS),
            "-i",
            "pipe:0",
            "-f",
            "ogg",
            "-acodec",
            "libopus",
            "pipe:1",
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )

    out, _ = proc.communicate(
        pcm.tobytes()
    )

    return out