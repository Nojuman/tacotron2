import librosa
import numpy as np
import hyperparams as hp


def ms_to_frames(ms, sample_rate):
    return int((ms / 1000) * sample_rate)


def load_wav(filename):
    return librosa.load(filename, sr=hp.sample_rate)


def wav_to_spectrogram(wav, sample_rate=hp.sample_rate,
                       fft_frame_size=hp.fft_frame_size,
                       fft_hop_size=hp.fft_hop_size,
                       num_mels=hp.num_mels,
                       min_freq=hp.min_freq,
                       max_freq=hp.max_freq,
                       floor_freq=hp.floor_freq):
    """
    Converts a wav file to a transposed db scale mel spectrogram.
    """
    n_fft = ms_to_frames(fft_frame_size, sample_rate)
    hop_length = ms_to_frames(fft_hop_size, sample_rate)
    mel_spec = librosa.feature.melspectrogram(wav, sr=sample_rate,
                                              n_fft=n_fft,
                                              hop_length=hop_length,
                                              n_mels=num_mels,
                                              fmin=min_freq,
                                              fmax=max_freq)
    return librosa.power_to_db(mel_spec, ref=floor_freq).T


def mu_law_encode(signal, mu=256):
    """
    Quantizes a signal to mu number discrete values.
    """
    mu = mu-1
    fx = np.sign(signal) * (np.log(1 + mu*np.abs(signal))/
                           np.log(1 + mu))
    return np.floor((fx+1)/2*mu+0.5).astype(np.long)


def decode_mu_law(quantized_signal, mu=256):
    mu = mu-1
    fx = (quantized_signal-0.5)/mu*2-1
    x = (np.sign(fx)/
         mu*((1+mu)**np.abs(fx)-1))
    return x