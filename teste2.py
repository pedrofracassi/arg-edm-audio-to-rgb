import numpy as np
from scipy.io import wavfile
from scipy.fftpack import fft
import matplotlib.pyplot as plt


def analyze_chunk(data, sample_rate, chunk_start_time):
    """
    Analyze a single chunk of audio data to find the three most prominent frequencies.

    Args:
        data (np.array): Chunk of audio data
        sample_rate (int): Sample rate of the audio
        chunk_start_time (float): Start time of the chunk in seconds

    Returns:
        list: Three most prominent frequencies in Hz
    """
    # Apply Hanning window
    window = np.hanning(len(data))
    data = data * window

    # Perform FFT
    fft_data = fft(data)
    freq_array = np.abs(fft_data)

    # Get frequency axis
    freq_axis = np.fft.fftfreq(len(freq_array), 1 / sample_rate)

    # Only look at positive frequencies
    positive_mask = freq_axis > 0
    freq_axis = freq_axis[positive_mask]
    freq_array = freq_array[positive_mask]

    # Find peaks (excluding very low frequencies)
    min_freq = 20  # Minimum frequency to consider (Hz)
    min_freq_idx = int(min_freq * len(freq_axis) / (sample_rate / 2))

    # Get indices of the three highest peaks
    peak_indices = np.argsort(freq_array[min_freq_idx:])[-3:][::-1] + min_freq_idx
    frequencies = freq_axis[peak_indices]
    magnitudes = freq_array[peak_indices]

    return frequencies.tolist(), magnitudes.tolist()


def analyze_frequencies_chunked(wav_path, chunk_duration=0.5):
    """
    Analyze a WAV file in chunks to find the three most prominent frequencies in each chunk.

    Args:
        wav_path (str): Path to the WAV file
        chunk_duration (float): Duration of each chunk in seconds

    Returns:
        dict: Analysis results including timestamps and frequencies
    """
    # Read the wav file
    sample_rate, data = wavfile.read(wav_path)

    # Convert stereo to mono if necessary
    if len(data.shape) > 1:
        data = np.mean(data, axis=1)

    # Calculate chunk size in samples
    chunk_size = int(chunk_duration * sample_rate)
    num_chunks = len(data) // chunk_size

    # Initialize results
    results = {"timestamps": [], "frequencies": [], "magnitudes": []}

    # Analyze each chunk
    for i in range(num_chunks):
        print(f"Analyzing chunk {i + 1}/{num_chunks}...")

        chunk_start = i * chunk_size
        chunk_end = (i + 1) * chunk_size
        chunk = data[chunk_start:chunk_end]

        chunk_time = i * chunk_duration
        frequencies, magnitudes = analyze_chunk(chunk, sample_rate, chunk_time)

        results["timestamps"].append(chunk_time)
        results["frequencies"].append(frequencies)
        results["magnitudes"].append(magnitudes)

    # Visualize the results
    # plot_frequency_evolution(results)

    return results


def plot_frequency_evolution(results):
    """
    Create a visualization of how frequencies change over time.
    """
    plt.figure(figsize=(15, 8))

    # Plot each frequency component
    colors = ["r", "g", "b"]
    labels = ["Highest", "Second", "Third"]

    for freq_idx in range(3):
        frequencies = [chunk[freq_idx] for chunk in results["frequencies"]]
        magnitudes = [chunk[freq_idx] for chunk in results["magnitudes"]]

        # Size of scatter points proportional to magnitude
        sizes = np.array(magnitudes) / max(magnitudes) * 100

        plt.scatter(
            results["timestamps"],
            frequencies,
            c=colors[freq_idx],
            s=sizes,
            alpha=0.6,
            label=f"{labels[freq_idx]} Peak",
        )

    plt.xlabel("Time (seconds)")
    plt.ylabel("Frequency (Hz)")
    plt.title("Frequency Evolution Over Time")
    plt.grid(True)
    plt.legend()
    plt.show()


import csv

# Example usage
if __name__ == "__main__":
    wav_file = "id.wav"  # Replace with your WAV file path
    results = analyze_frequencies_chunked(wav_file)

    # Print results for each chunk
    for i, (time, freqs) in enumerate(
        zip(results["timestamps"], results["frequencies"])
    ):
        print(f"\nChunk at {time:.1f}s:")
        for j, freq in enumerate(freqs, 1):
            print(f"{j}. {freq:.1f} Hz")

    # Save the results as a CSV file
    with open("frequency_results.csv", mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            ["Time (s)", "Frequency 1 (Hz)", "Frequency 2 (Hz)", "Frequency 3 (Hz)"]
        )
        for time, freqs in zip(results["timestamps"], results["frequencies"]):
            writer.writerow([time] + sorted(freqs))
