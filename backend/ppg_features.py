import numpy as np

def extract_ppg_features(signal):
    """
    Extract 11 features from a PPG segment (list or array).
    Features:
        0. Mean
        1. Standard deviation
        2. Skewness
        3. Kurtosis
        4. Signal range (max - min)
        5. Zero-crossing rate (number of sign changes in detrended signal)
        6. RMS (root mean square)
        7. Peak-to-peak (same as range)
        8. Spectral centroid (weighted mean of frequency components)
        9. Spectral entropy (normalized Shannon entropy of power spectrum)
        10. Autocorrelation peak lag (first local maximum of autocorrelation, >= lag 2)
    """
    arr = np.asarray(signal, dtype=float)
    if len(arr) < 2:
        return np.zeros(11)  # fallback if segment too short

    # --- Statistical features (original 8) ---
    mean = np.mean(arr)
    std = np.std(arr)
    if std < 1e-7:
        std = 1e-7

    diff = arr - mean
    skewness = np.mean(np.power(diff, 3)) / (np.power(std, 3) + 1e-7)
    kurtosis = np.mean(np.power(diff, 4)) / (np.power(std, 4) + 1e-7)
    signal_range = np.max(arr) - np.min(arr)
    # Zero-crossings of detrended signal
    zero_crossings = np.sum(np.abs(np.diff(np.sign(diff)))) // 2
    rms = np.sqrt(np.mean(np.square(arr)))
    peak_to_peak = signal_range

    # --- Spectral features ---
    n = len(arr)
    if n > 1:
        # FFT
        fft_vals = np.fft.rfft(arr)
        power = np.abs(fft_vals) ** 2
        freqs = np.fft.rfftfreq(n, d=1.0)  # assume unit spacing (samples)

        # Spectral centroid (weighted mean of frequencies)
        if np.sum(power) > 0:
            centroid = np.sum(freqs * power) / np.sum(power)
        else:
            centroid = 0.0

        # Spectral entropy (normalized Shannon entropy)
        power_norm = power / (np.sum(power) + 1e-12)
        entropy = -np.sum(power_norm * np.log2(power_norm + 1e-12))
        # Normalize by log2(number of frequency bins)
        max_entropy = np.log2(len(power_norm))
        if max_entropy > 0:
            spectral_entropy = entropy / max_entropy
        else:
            spectral_entropy = 0.0
    else:
        centroid = 0.0
        spectral_entropy = 0.0

    # --- Autocorrelation features ---
    # Compute autocorrelation up to half the signal length
    if n > 3:
        autocorr = np.correlate(arr, arr, mode='full')
        autocorr = autocorr[len(autocorr)//2:]  # keep only non-negative lags
        # Find first local maximum after lag 1 (ignore lag 0 which is always peak)
        if len(autocorr) > 2:
            # Find peaks where value > neighbors
            peaks = []
            for i in range(1, len(autocorr)-1):
                if autocorr[i] > autocorr[i-1] and autocorr[i] > autocorr[i+1]:
                    peaks.append(i)
            # Choose the first peak with lag >= 2 (to avoid trivial)
            first_peak_lag = None
            for p in peaks:
                if p >= 2:
                    first_peak_lag = p
                    break
            autocorr_peak_lag = first_peak_lag if first_peak_lag is not None else 0
        else:
            autocorr_peak_lag = 0
    else:
        autocorr_peak_lag = 0

    # Return all 11 features
    return np.array([
        mean, std, skewness, kurtosis,
        signal_range, zero_crossings,
        rms, peak_to_peak,
        centroid, spectral_entropy, autocorr_peak_lag
    ], dtype=float)
