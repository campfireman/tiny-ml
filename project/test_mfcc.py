import librosa
import wave
import numpy as np
from python_speech_features import mfcc
from scipy.fftpack import dct
import speechpy


def custom(y, sr):
    # 1) mel‐spectrogram with NO center‐padding
    S = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=2048,
        hop_length=512,
        n_mels=40,
        fmin=0,
        fmax=sr/2,
        power=2.0,
        htk=True,
        center=False          # ← crucial
    )

    # 2) log
    log_S = np.log(S + np.finfo(float).eps)

    # 3) un‐normalized DCT‐II (type=2, norm=None)
    #    shape: (n_mels, n_frames)
    mfcc_no_norm = dct(log_S, type=2, axis=0, norm=None)

    # 4) take the first 12 cepstral rows, transpose → (n_frames, 12)
    mfcc_12 = mfcc_no_norm[:12, :].T

    # 5) build matching log‐energy vector from exactly the same framing
    frames = librosa.util.frame(
        y,
        frame_length=2048,
        hop_length=512
        # default: no padding
    )
    log_energy = np.log((frames**2).sum(axis=0) + np.finfo(float).eps)

    # now both have shape (n_frames, …), so we can concatenate
    return np.hstack([mfcc_12, log_energy[:, None]])


def main():
    file = './data/2025-05-27_20-49-15-chillaxo-close.wav'
    with wave.open(file, 'rb') as wav:
        n_channels = wav.getnchannels()
        sample_width = wav.getsampwidth()  # in bytes
        n_frames = wav.getnframes()
        raw_bytes = wav.readframes(n_frames)

        # Convert to numpy array
        dtype = {1: np.int8, 2: np.int16, 4: np.int32}[sample_width]
        data = np.frombuffer(raw_bytes, dtype=dtype)

        # Reshape if stereo
        if n_channels > 1:
            data = data.reshape(-1, n_channels)

        print(f"First 10 amplitudes: {data.shape}")
        np.savetxt('amplitudes.txt', data, fmt='%d')

        yy, sr = librosa.load(file, sr=None)
        print(sr)
        # mfccs = librosa.feature.mfcc(
        #     y=yy,
        #     sr=sr,
        #     n_mfcc=13,
        #     n_fft=2048,
        #     hop_length=512,
        #     n_mels=40,
        #     fmin=0,
        #     fmax=sr/2,
        #     lifter=1,
        #     center=False
        # ).T
        # mfccs = mfcc(
        #     signal=yy,
        #     samplerate=sr,
        #     winlen=0.128,
        #     winstep=0.032,
        #     numcep=13,
        #     nfilt=40,
        #     nfft=2048,
        #     lowfreq=0,
        #     highfreq=sr//2,
        #     ceplifter=1,
        #     appendEnergy=True
        # )
        frame_length = 2048 / sr  # 0.128 s
        frame_stride = 512 / sr  # 0.032 s

        mfccs = speechpy.feature.mfcc(
            yy,
            sampling_frequency=sr,
            frame_length=frame_length,
            frame_stride=frame_stride,
            num_cepstral=13,
            num_filters=40,
            fft_length=2048,
            low_frequency=0,
            high_frequency=None,
            dc_elimination=True
        )
        # mfccs = custom(yy, sr)
        print(mfccs.shape)
        np.savetxt('mfcc.txt', mfccs, fmt='%f')


if __name__ == "__main__":

    main()
