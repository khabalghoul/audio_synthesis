#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import jack
import librosa
import numpy as np
import socket

buf = np.zeros(0, np.float32)

HOST, PORT = "localhost", 65431
GAIN = 5e-2  # escala latente
SILENCE = 1e-4  # umbral de silencio (~-80 dB FS)

# ---------- JACK ----------
client = jack.Client("Audio2Latent")
in_l = client.inports.register("in_l")
in_r = client.inports.register("in_r")

# ---------- TCP ----------
tcp = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
tcp.connect((HOST, PORT))
print("✅ Conectado a", (HOST, PORT))

BOOST = 5

SMOOTH = 0.8                # 0→sin suavizar, 0.99→muy liso
prev_z = np.zeros(512, np.float32)

BLOCK = 512  # 1024 muestras ≈ 21 ms a 48 kHz
BANDS = 8
# ATTACK  = 1 - np.exp(-1/160)    #  4 ms
# RELEASE = 1 - np.exp(-1/8000)   # 170 ms
ATTACK  = 1 - np.exp(-1/40)     # 1,6 ms  (antes 4 ms)
RELEASE = 1 - np.exp(-1/2000)   # 83 ms   (antes 170 ms)
GAIN_BAND = 0.07
GAIN_WALK = 0.001
ROLL_STEPS = 300
SILENCE = 1e-4          # umbral RMS para “silencio”

mel_fb = librosa.filters.mel(
    sr=client.samplerate,
    n_fft=BLOCK * 2,
    n_mels=BANDS,
    fmin=40,
    fmax=16000).astype(np.float32)

env = np.zeros(BANDS, np.float32)  # envolventes por banda
R = np.random.randn(BANDS, 512).astype(np.float32)
R /= np.linalg.norm(R, axis=1, keepdims=True)
walk = np.zeros(512, np.float32)
frame_count = 0


def rms(x):  # módulo → siempre float
    return np.sqrt(np.mean(np.abs(x) ** 2) + 1e-12)


@client.set_process_callback
def process(frames):
    global env, walk, frame_count, R, buf

    l = np.frombuffer(in_l.get_buffer(), dtype=np.float32, count=frames)
    r = np.frombuffer(in_r.get_buffer(), dtype=np.float32, count=frames)
    mono = 0.5 * (l + r)

    for off in range(0, frames, 64):          # 64 = tamaño real de JACK
        #buf = np.concatenate((buf, mono[off: off+64]))
        buf = np.concatenate((buf, mono[:frames]))

        while len(buf) >= BLOCK:              # tenemos 1024 muestras
            chunk, buf = buf[:BLOCK], buf[BLOCK:]

            # -- RMS global para puerta de silencio --
            amp = np.sqrt(np.mean(chunk**2) + 1e-12)

            if amp < SILENCE:
                walk *= 0.98
                env *= 0.995  # decay pasivo para no dejarla clavada
                z = np.zeros(512, np.float32)
            else:
                # ---------- magnitud en bandas ----------
                spectrum = np.abs(np.fft.rfft(chunk * np.hanning(BLOCK), n=BLOCK*2))

                # nos da un vector con la energia en cada banda mel
                band_energy = mel_fb @ spectrum

                # ---------- envolventes A/D ----------
                up = band_energy > env
                env[up]  += (band_energy[up]  - env[up])  * ATTACK
                env[~up] += (band_energy[~up] - env[~up]) * RELEASE

                # ---------- empuje por bandas ----------
                delta_z = (GAIN_BAND * env / (amp + 1e-6)) @ R
                # amp_log = np.log10(amp + 1e-6)  # -inf … 0
                # norm = np.clip((amp_log + 4) / 4, 0, 1)  # 0 a 1 para -40 dB…0 dB
                # delta_z = (GAIN_BAND / (norm + 0.3)) * env @ R

                # ---------- random-walk + rolling ----------
                walk += GAIN_WALK * np.random.randn(512).astype(np.float32)
                walk *= 0.999
                frame_count += 1
                if frame_count % ROLL_STEPS == 0:
                    walk = np.roll(walk, 1)

                z = np.clip(walk + delta_z, -1, 1).astype(np.float32)
                z *= BOOST

                # z_raw = np.clip(walk + delta_z, -1, 1).astype(np.float32)
                # global prev_z
                # z = SMOOTH * prev_z + (1 - SMOOTH) * z_raw  # low-pass 1er orden
                prev_z = z

            # ---------- enviar (2 KiB) ----------
            try:
                tcp.sendall(z.tobytes())
            except BrokenPipeError:
                print("⚠️ Conexión perdida. Latentes no enviados.")

            # Muestra 4 valores para depurar
            #print("snd >", z[:4], len(z))

with client:  # activa el puerto
    try:
        client.connect("system:capture_1", "Audio2Latent:in_l")
        client.connect("system:capture_2", "Audio2Latent:in_r")
    except jack.JackError:
        pass
    input("⏯  Grabando…  Pulsa <Enter> para salir.\n")

tcp.close()
