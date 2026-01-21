#!/usr/bin/env python3
"""
Vortex Shedding Frequency Analyzer
==================================
Analyzes wind tunnel video to detect vortex shedding frequency by tracking
brightness oscillations in a defined region of interest (ROI), then computes
the Strouhal number.

Author: MAE 157W Lab 1
"""

import cv2
import numpy as np
from scipy import signal
from scipy.fft import rfft, rfftfreq
import matplotlib.pyplot as plt
from pathlib import Path


# =============================================================================
# CONFIGURATION - Modify these parameters for your experiment
# =============================================================================

# Video settings
VIDEO_PATH = "Laser1.mov"  # Path to your wind tunnel video

# Region of Interest (ROI) - (x, y, width, height) in pixels
# Set these coordinates to capture the wake region behind the object
ROI_X = 950       # X coordinate of top-left corner
ROI_Y = 500       # Y coordinate of top-left corner
ROI_WIDTH = 50    # Width of the ROI
ROI_HEIGHT = 50   # Height of the ROI

# Physical parameters for Strouhal number calculation
# St = f * L / V
CHARACTERISTIC_LENGTH = 0.01  # meters (e.g., cylinder diameter)
FLOW_VELOCITY = 10.0          # m/s (free stream velocity)

# Analysis settings
FRAME_RATE = None        # Set to None to auto-detect from video, or override (fps)
BANDPASS_LOW = 1.0       # Hz - lower cutoff (filters out DC drift)
BANDPASS_HIGH = 100.0    # Hz - upper cutoff (filters out high-freq noise)
USE_BANDPASS = True      # Enable/disable bandpass filtering


# =============================================================================
# ANALYSIS FUNCTIONS
# =============================================================================

def load_video_info(video_path: str) -> tuple:
    """
    Load video and extract metadata.
    
    Returns:
        cap: VideoCapture object
        fps: Frame rate
        total_frames: Total number of frames
        frame_width: Video width in pixels
        frame_height: Video height in pixels
    """
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video file: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    return cap, fps, total_frames, frame_width, frame_height


def extract_roi_brightness(cap, roi: tuple, total_frames: int) -> tuple:
    """
    Extract mean brightness from ROI for each frame.
    
    Args:
        cap: VideoCapture object
        roi: Tuple of (x, y, width, height)
        total_frames: Number of frames to process
        
    Returns:
        brightness: Array of mean brightness values
        sample_frame: First frame for visualization
        roi_frame: First frame with ROI overlay
    """
    x, y, w, h = roi
    brightness = []
    sample_frame = None
    roi_frame = None
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset to start
    
    print(f"Extracting brightness from {total_frames} frames...")
    
    for i in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Extract ROI and compute mean brightness
        roi_region = gray[y:y+h, x:x+w]
        mean_brightness = np.mean(roi_region)
        brightness.append(mean_brightness)
        
        # Store first frame for visualization
        if i == 0:
            sample_frame = gray.copy()
            # Create color version with ROI overlay
            roi_frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            cv2.rectangle(roi_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(roi_frame, "ROI", (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Progress indicator
        if (i + 1) % 100 == 0 or i == total_frames - 1:
            print(f"  Processed {i + 1}/{total_frames} frames")
    
    return np.array(brightness), sample_frame, roi_frame


def preprocess_signal(brightness: np.ndarray, fps: float, 
                      low_freq: float, high_freq: float,
                      use_bandpass: bool = True) -> np.ndarray:
    """
    Preprocess the brightness signal: detrend and optionally bandpass filter.
    
    Args:
        brightness: Raw brightness time series
        fps: Sampling frequency (frames per second)
        low_freq: Lower cutoff frequency (Hz)
        high_freq: Upper cutoff frequency (Hz)
        use_bandpass: Whether to apply bandpass filter
        
    Returns:
        Preprocessed signal
    """
    # Remove linear trend (DC offset and drift)
    detrended = signal.detrend(brightness, type='linear')
    
    if use_bandpass and fps > 2 * high_freq:
        # Design Butterworth bandpass filter
        nyquist = fps / 2
        low = low_freq / nyquist
        high = min(high_freq / nyquist, 0.99)  # Ensure < 1
        
        if low < high and low > 0:
            b, a = signal.butter(4, [low, high], btype='band')
            filtered = signal.filtfilt(b, a, detrended)
            return filtered
    
    return detrended


def compute_fft(signal_data: np.ndarray, fps: float) -> tuple:
    """
    Compute FFT of the signal to find frequency content.
    
    Args:
        signal_data: Preprocessed time series
        fps: Sampling frequency
        
    Returns:
        frequencies: Array of frequencies (Hz)
        power: Power spectrum (magnitude squared)
    """
    n = len(signal_data)
    
    # Apply window function to reduce spectral leakage
    window = np.hanning(n)
    windowed_signal = signal_data * window
    
    # Compute real FFT (signal is real-valued)
    fft_result = rfft(windowed_signal)
    frequencies = rfftfreq(n, d=1/fps)
    
    # Compute power spectrum (magnitude squared)
    power = np.abs(fft_result) ** 2
    
    # Normalize
    power = power / np.max(power)
    
    return frequencies, power


def find_dominant_frequency(frequencies: np.ndarray, power: np.ndarray,
                            min_freq: float = 0.5) -> tuple:
    """
    Find the dominant frequency (highest peak) in the power spectrum.
    
    Args:
        frequencies: Frequency array
        power: Power spectrum
        min_freq: Minimum frequency to consider (ignore DC component)
        
    Returns:
        peak_freq: Dominant frequency (Hz)
        peak_power: Power at dominant frequency
        all_peaks: Indices of all detected peaks
    """
    # Find peaks in the power spectrum
    # Only consider frequencies above min_freq
    valid_idx = frequencies > min_freq
    valid_freqs = frequencies[valid_idx]
    valid_power = power[valid_idx]
    
    # Find peaks with minimum prominence
    peaks, properties = signal.find_peaks(valid_power, 
                                          prominence=0.1,
                                          distance=5)
    
    if len(peaks) == 0:
        # No clear peaks found, use maximum
        max_idx = np.argmax(valid_power)
        return valid_freqs[max_idx], valid_power[max_idx], []
    
    # Find the highest peak
    peak_powers = valid_power[peaks]
    dominant_idx = peaks[np.argmax(peak_powers)]
    
    peak_freq = valid_freqs[dominant_idx]
    peak_power = valid_power[dominant_idx]
    
    # Convert peak indices back to original array indexing
    original_peaks = np.where(valid_idx)[0][peaks]
    
    return peak_freq, peak_power, original_peaks


def compute_autocorrelation(signal_data: np.ndarray, fps: float) -> tuple:
    """
    Compute the autocorrelation of the signal to find periodic patterns.
    
    Autocorrelation measures the similarity of a signal with a delayed copy
    of itself. For periodic signals, peaks in the autocorrelation correspond
    to the period of oscillation.
    
    Args:
        signal_data: Preprocessed time series
        fps: Sampling frequency (frames per second)
        
    Returns:
        lags: Time lag array (seconds)
        autocorr: Normalized autocorrelation values
    """
    n = len(signal_data)
    
    # Normalize signal (zero mean, unit variance)
    sig_normalized = (signal_data - np.mean(signal_data)) / np.std(signal_data)
    
    # Compute full autocorrelation using numpy correlate
    autocorr_full = np.correlate(sig_normalized, sig_normalized, mode='full')
    
    # Take only the positive lags (second half)
    autocorr = autocorr_full[n-1:]
    
    # Normalize by the zero-lag value (which equals n for normalized signal)
    autocorr = autocorr / autocorr[0]
    
    # Create lag array in seconds
    lags = np.arange(len(autocorr)) / fps
    
    return lags, autocorr


def find_autocorr_frequency(lags: np.ndarray, autocorr: np.ndarray, 
                            fps: float, min_freq: float = 0.5) -> tuple:
    """
    Find the dominant frequency from autocorrelation peaks.
    
    The first significant peak after lag=0 corresponds to the fundamental
    period of the signal.
    
    Args:
        lags: Time lag array (seconds)
        autocorr: Normalized autocorrelation values
        fps: Sampling frequency
        min_freq: Minimum frequency to consider (Hz)
        
    Returns:
        peak_freq: Dominant frequency from autocorrelation (Hz)
        peak_period: Period corresponding to dominant frequency (seconds)
        peak_indices: Indices of all detected peaks
        peak_lags: Lag values at detected peaks (seconds)
    """
    # Minimum lag corresponding to maximum frequency we care about
    max_period = 1.0 / min_freq  # Maximum period in seconds
    min_lag_samples = int(fps / (BANDPASS_HIGH if USE_BANDPASS else 100))  # Skip very short lags
    
    # Only look at lags beyond the initial decay and up to max_period
    max_lag_samples = min(int(max_period * fps * 2), len(autocorr) - 1)
    
    # Find peaks in the autocorrelation
    # Start search after min_lag_samples to avoid the central peak
    search_start = max(min_lag_samples, 3)
    search_autocorr = autocorr[search_start:max_lag_samples]
    
    # Find peaks with minimum height and distance
    peaks, properties = signal.find_peaks(
        search_autocorr,
        height=0.1,  # Minimum correlation of 0.1
        distance=int(fps * 0.01),  # Minimum distance between peaks
        prominence=0.05
    )
    
    if len(peaks) == 0:
        # No clear peaks found, return estimate from first zero crossing
        zero_crossings = np.where(np.diff(np.sign(autocorr[search_start:max_lag_samples])))[0]
        if len(zero_crossings) >= 2:
            # Estimate period from zero crossings (half period between consecutive crossings)
            half_period_samples = zero_crossings[1] - zero_crossings[0]
            period_samples = 2 * half_period_samples + search_start
            period = period_samples / fps
            return 1.0 / period, period, [], []
        return 0.0, 0.0, [], []
    
    # Adjust peak indices to account for search_start offset
    peaks_adjusted = peaks + search_start
    
    # The first significant peak gives the fundamental period
    first_peak_idx = peaks_adjusted[0]
    peak_period = lags[first_peak_idx]
    peak_freq = 1.0 / peak_period if peak_period > 0 else 0.0
    
    # Get lag values at all peaks
    peak_lags = lags[peaks_adjusted]
    
    return peak_freq, peak_period, peaks_adjusted, peak_lags


def calculate_strouhal(frequency: float, length: float, velocity: float) -> float:
    """
    Calculate the Strouhal number.
    
    St = f * L / V
    
    Args:
        frequency: Vortex shedding frequency (Hz)
        length: Characteristic length (m)
        velocity: Flow velocity (m/s)
        
    Returns:
        Strouhal number (dimensionless)
    """
    if velocity <= 0:
        raise ValueError("Flow velocity must be positive")
    
    return (frequency * length) / velocity


def create_visualization(roi_frame: np.ndarray, 
                        time: np.ndarray, 
                        brightness: np.ndarray,
                        processed_signal: np.ndarray,
                        frequencies: np.ndarray, 
                        power: np.ndarray,
                        peak_freq_fft: float,
                        strouhal_fft: float,
                        peaks_fft: np.ndarray,
                        lags: np.ndarray,
                        autocorr: np.ndarray,
                        peak_freq_autocorr: float,
                        peak_period_autocorr: float,
                        strouhal_autocorr: float,
                        peaks_autocorr: np.ndarray) -> plt.Figure:
    """
    Create a multi-panel visualization of the analysis results.
    Includes both FFT and Autocorrelation analysis for comparison.
    """
    fig = plt.figure(figsize=(16, 14))
    fig.suptitle('Vortex Shedding Frequency Analysis\n(FFT vs Autocorrelation Comparison)', 
                 fontsize=14, fontweight='bold')
    
    # Create grid layout - 4 rows, 2 columns
    gs = fig.add_gridspec(4, 2, height_ratios=[1, 1, 1.2, 1.2], hspace=0.4, wspace=0.25)
    
    # Panel 1: Frame with ROI
    ax1 = fig.add_subplot(gs[0, 0])
    if roi_frame is not None:
        ax1.imshow(cv2.cvtColor(roi_frame, cv2.COLOR_BGR2RGB))
    ax1.set_title('Video Frame with ROI', fontweight='bold')
    ax1.set_xlabel('X (pixels)')
    ax1.set_ylabel('Y (pixels)')
    
    # Panel 2: Raw brightness time series
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(time, brightness, 'b-', linewidth=0.5, alpha=0.7)
    ax2.set_title('Raw Brightness Signal', fontweight='bold')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Mean Brightness (0-255)')
    ax2.grid(True, alpha=0.3)
    
    # Panel 3: Processed signal (full width)
    ax3 = fig.add_subplot(gs[1, :])
    ax3.plot(time, processed_signal, 'g-', linewidth=0.5)
    ax3.set_title('Preprocessed Signal (Detrended & Filtered)', fontweight='bold')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Brightness Deviation')
    ax3.grid(True, alpha=0.3)
    
    # Panel 4: FFT Power Spectrum (full width)
    ax4 = fig.add_subplot(gs[2, :])
    ax4.semilogy(frequencies, power, 'b-', linewidth=1)
    
    # Mark FFT peaks
    if len(peaks_fft) > 0:
        ax4.semilogy(frequencies[peaks_fft], power[peaks_fft], 'r^', markersize=8, 
                     label='Detected peaks')
    
    # Highlight dominant frequency
    ax4.axvline(x=peak_freq_fft, color='r', linestyle='--', linewidth=2, 
                label=f'Dominant: {peak_freq_fft:.2f} Hz')
    
    ax4.set_title('FFT Power Spectrum', fontweight='bold')
    ax4.set_xlabel('Frequency (Hz)')
    ax4.set_ylabel('Normalized Power')
    ax4.set_xlim([0, min(frequencies[-1], BANDPASS_HIGH * 2)])
    ax4.legend(loc='upper right')
    ax4.grid(True, alpha=0.3)
    
    # Add FFT results text box
    fft_text = (
        f"FFT Results\n"
        f"{'─' * 25}\n"
        f"Frequency: {peak_freq_fft:.3f} Hz\n"
        f"Period: {1000/peak_freq_fft:.2f} ms\n"
        f"Strouhal: {strouhal_fft:.4f}"
    )
    props = dict(boxstyle='round', facecolor='lightblue', alpha=0.8)
    ax4.text(0.02, 0.98, fft_text, transform=ax4.transAxes, fontsize=9,
             verticalalignment='top', horizontalalignment='left',
             bbox=props, fontfamily='monospace')
    
    # Panel 5: Autocorrelation (full width)
    ax5 = fig.add_subplot(gs[3, :])
    
    # Limit display to reasonable lag range
    max_display_lag = min(2.0, lags[-1])  # Show up to 2 seconds or max available
    display_mask = lags <= max_display_lag
    
    ax5.plot(lags[display_mask], autocorr[display_mask], 'purple', linewidth=1)
    ax5.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
    
    # Mark autocorrelation peaks
    if len(peaks_autocorr) > 0:
        valid_peaks = peaks_autocorr[peaks_autocorr < len(lags[display_mask])]
        if len(valid_peaks) > 0:
            ax5.plot(lags[valid_peaks], autocorr[valid_peaks], 'r^', markersize=8,
                     label='Detected peaks')
    
    # Highlight first peak (fundamental period)
    if peak_period_autocorr > 0:
        ax5.axvline(x=peak_period_autocorr, color='r', linestyle='--', linewidth=2,
                    label=f'Period: {peak_period_autocorr*1000:.2f} ms')
    
    ax5.set_title('Autocorrelation Analysis', fontweight='bold')
    ax5.set_xlabel('Lag (seconds)')
    ax5.set_ylabel('Autocorrelation')
    ax5.set_xlim([0, max_display_lag])
    ax5.set_ylim([-0.5, 1.1])
    ax5.legend(loc='upper right')
    ax5.grid(True, alpha=0.3)
    
    # Add Autocorrelation results text box
    autocorr_text = (
        f"Autocorrelation Results\n"
        f"{'─' * 25}\n"
        f"Frequency: {peak_freq_autocorr:.3f} Hz\n"
        f"Period: {peak_period_autocorr*1000:.2f} ms\n"
        f"Strouhal: {strouhal_autocorr:.4f}"
    )
    props = dict(boxstyle='round', facecolor='plum', alpha=0.8)
    ax5.text(0.02, 0.98, autocorr_text, transform=ax5.transAxes, fontsize=9,
             verticalalignment='top', horizontalalignment='left',
             bbox=props, fontfamily='monospace')
    
    # Add comparison summary text box
    freq_diff = abs(peak_freq_fft - peak_freq_autocorr)
    freq_diff_pct = 100 * freq_diff / peak_freq_fft if peak_freq_fft > 0 else 0
    
    comparison_text = (
        f"Comparison Summary\n"
        f"{'═' * 30}\n"
        f"FFT Frequency:    {peak_freq_fft:.3f} Hz\n"
        f"Autocorr Freq:    {peak_freq_autocorr:.3f} Hz\n"
        f"Difference:       {freq_diff:.3f} Hz ({freq_diff_pct:.1f}%)\n"
        f"{'─' * 30}\n"
        f"Characteristic L: {CHARACTERISTIC_LENGTH*1000:.2f} mm\n"
        f"Flow Velocity:    {FLOW_VELOCITY:.2f} m/s"
    )
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.9)
    ax5.text(0.98, 0.98, comparison_text, transform=ax5.transAxes, fontsize=9,
             verticalalignment='top', horizontalalignment='right',
             bbox=props, fontfamily='monospace')
    
    plt.tight_layout()
    return fig


def print_results(peak_freq_fft: float, strouhal_fft: float,
                  peak_freq_autocorr: float, peak_period_autocorr: float,
                  strouhal_autocorr: float,
                  fps: float, total_frames: int, duration: float):
    """Print analysis results to console."""
    print("\n" + "=" * 60)
    print("VORTEX SHEDDING ANALYSIS RESULTS")
    print("=" * 60)
    print(f"\nVideo Properties:")
    print(f"  • Frame Rate: {fps:.2f} fps")
    print(f"  • Total Frames: {total_frames}")
    print(f"  • Duration: {duration:.2f} seconds")
    print(f"\nPhysical Parameters:")
    print(f"  • Characteristic Length (L): {CHARACTERISTIC_LENGTH*1000:.2f} mm")
    print(f"  • Flow Velocity (V): {FLOW_VELOCITY:.2f} m/s")
    
    print(f"\n" + "-" * 60)
    print("METHOD 1: FFT Analysis")
    print("-" * 60)
    print(f"  • Vortex Shedding Frequency (f): {peak_freq_fft:.3f} Hz")
    print(f"  • Period: {1/peak_freq_fft*1000:.2f} ms")
    print(f"  • Strouhal Number (St = fL/V): {strouhal_fft:.4f}")
    
    print(f"\n" + "-" * 60)
    print("METHOD 2: Autocorrelation Analysis")
    print("-" * 60)
    print(f"  • Vortex Shedding Frequency (f): {peak_freq_autocorr:.3f} Hz")
    print(f"  • Period: {peak_period_autocorr*1000:.2f} ms")
    print(f"  • Strouhal Number (St = fL/V): {strouhal_autocorr:.4f}")
    
    # Comparison
    freq_diff = abs(peak_freq_fft - peak_freq_autocorr)
    freq_diff_pct = 100 * freq_diff / peak_freq_fft if peak_freq_fft > 0 else 0
    
    print(f"\n" + "-" * 60)
    print("COMPARISON")
    print("-" * 60)
    print(f"  • Frequency Difference: {freq_diff:.3f} Hz ({freq_diff_pct:.1f}%)")
    avg_freq = (peak_freq_fft + peak_freq_autocorr) / 2
    avg_strouhal = (strouhal_fft + strouhal_autocorr) / 2
    print(f"  • Average Frequency: {avg_freq:.3f} Hz")
    print(f"  • Average Strouhal Number: {avg_strouhal:.4f}")
    
    print("\n" + "=" * 60)
    
    # Compare to theoretical values
    print("\nReference: For a circular cylinder, St ≈ 0.2 at Re > 1000")
    if 0.15 < avg_strouhal < 0.25:
        print("Your average Strouhal number is in the expected range for cylinder wake!")
    print("=" * 60 + "\n")


def main():
    """Main analysis pipeline."""
    print("\n" + "=" * 60)
    print("VORTEX SHEDDING FREQUENCY ANALYZER")
    print("=" * 60)
    
    # Validate video path
    if not Path(VIDEO_PATH).exists():
        print(f"\nERROR: Video file not found: {VIDEO_PATH}")
        print("Please update VIDEO_PATH in the configuration section.")
        print("\nTo use this script:")
        print("  1. Set VIDEO_PATH to your wind tunnel video file")
        print("  2. Adjust ROI coordinates to capture the wake region")
        print("  3. Set CHARACTERISTIC_LENGTH and FLOW_VELOCITY")
        print("  4. Run the script again")
        return
    
    # Load video
    print(f"\nLoading video: {VIDEO_PATH}")
    cap, fps, total_frames, width, height = load_video_info(VIDEO_PATH)
    
    # Use configured frame rate if specified
    if FRAME_RATE is not None:
        fps = FRAME_RATE
        print(f"Using configured frame rate: {fps} fps")
    
    duration = total_frames / fps
    print(f"Video info: {width}x{height}, {fps:.2f} fps, {total_frames} frames, {duration:.2f}s")
    
    # Validate ROI
    roi = (ROI_X, ROI_Y, ROI_WIDTH, ROI_HEIGHT)
    if ROI_X + ROI_WIDTH > width or ROI_Y + ROI_HEIGHT > height:
        print(f"\nERROR: ROI exceeds frame boundaries!")
        print(f"Frame size: {width}x{height}")
        print(f"ROI: x={ROI_X}, y={ROI_Y}, w={ROI_WIDTH}, h={ROI_HEIGHT}")
        cap.release()
        return
    
    print(f"ROI: x={ROI_X}, y={ROI_Y}, width={ROI_WIDTH}, height={ROI_HEIGHT}")
    
    # Extract brightness from ROI
    brightness, sample_frame, roi_frame = extract_roi_brightness(cap, roi, total_frames)
    cap.release()
    
    # Create time array
    time = np.arange(len(brightness)) / fps
    
    # Preprocess signal
    print("\nPreprocessing signal...")
    processed_signal = preprocess_signal(
        brightness, fps, 
        BANDPASS_LOW, BANDPASS_HIGH,
        USE_BANDPASS
    )
    
    # Compute FFT
    print("Computing FFT...")
    frequencies, power = compute_fft(processed_signal, fps)
    
    # Find dominant frequency from FFT
    print("Finding dominant frequency (FFT)...")
    peak_freq_fft, peak_power_fft, peaks_fft = find_dominant_frequency(
        frequencies, power, 
        min_freq=BANDPASS_LOW
    )
    
    # Calculate Strouhal number from FFT
    strouhal_fft = calculate_strouhal(peak_freq_fft, CHARACTERISTIC_LENGTH, FLOW_VELOCITY)
    
    # Compute Autocorrelation
    print("Computing autocorrelation...")
    lags, autocorr = compute_autocorrelation(processed_signal, fps)
    
    # Find dominant frequency from autocorrelation
    print("Finding dominant frequency (Autocorrelation)...")
    peak_freq_autocorr, peak_period_autocorr, peaks_autocorr, peak_lags = find_autocorr_frequency(
        lags, autocorr, fps, min_freq=BANDPASS_LOW
    )
    
    # Calculate Strouhal number from autocorrelation
    strouhal_autocorr = calculate_strouhal(peak_freq_autocorr, CHARACTERISTIC_LENGTH, FLOW_VELOCITY) if peak_freq_autocorr > 0 else 0.0
    
    # Print results
    print_results(peak_freq_fft, strouhal_fft, 
                  peak_freq_autocorr, peak_period_autocorr, strouhal_autocorr,
                  fps, total_frames, duration)
    
    # Create visualization
    print("Generating plots...")
    fig = create_visualization(
        roi_frame, time, brightness, processed_signal,
        frequencies, power, peak_freq_fft, strouhal_fft, peaks_fft,
        lags, autocorr, peak_freq_autocorr, peak_period_autocorr, 
        strouhal_autocorr, peaks_autocorr
    )
    
    # Save figure
    output_path = Path(VIDEO_PATH).stem + "_analysis.png"
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Figure saved: {output_path}")
    
    # Show plot
    plt.show()
    
    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()

