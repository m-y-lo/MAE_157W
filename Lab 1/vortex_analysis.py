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
                        peak_freq: float,
                        strouhal: float,
                        peaks: np.ndarray) -> plt.Figure:
    """
    Create a multi-panel visualization of the analysis results.
    """
    fig = plt.figure(figsize=(14, 10))
    fig.suptitle('Vortex Shedding Frequency Analysis', fontsize=14, fontweight='bold')
    
    # Create grid layout
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1], hspace=0.35, wspace=0.25)
    
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
    
    # Panel 3: Processed signal
    ax3 = fig.add_subplot(gs[1, :])
    ax3.plot(time, processed_signal, 'g-', linewidth=0.5)
    ax3.set_title('Preprocessed Signal (Detrended & Filtered)', fontweight='bold')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Brightness Deviation')
    ax3.grid(True, alpha=0.3)
    
    # Panel 4: FFT Power Spectrum
    ax4 = fig.add_subplot(gs[2, :])
    ax4.semilogy(frequencies, power, 'b-', linewidth=1)
    
    # Mark peaks
    if len(peaks) > 0:
        ax4.semilogy(frequencies[peaks], power[peaks], 'r^', markersize=8, 
                     label='Detected peaks')
    
    # Highlight dominant frequency
    ax4.axvline(x=peak_freq, color='r', linestyle='--', linewidth=2, 
                label=f'Dominant: {peak_freq:.2f} Hz')
    
    ax4.set_title('FFT Power Spectrum', fontweight='bold')
    ax4.set_xlabel('Frequency (Hz)')
    ax4.set_ylabel('Normalized Power')
    ax4.set_xlim([0, min(frequencies[-1], BANDPASS_HIGH * 2)])
    ax4.legend(loc='upper right')
    ax4.grid(True, alpha=0.3)
    
    # Add results text box
    results_text = (
        f"Results Summary\n"
        f"{'─' * 30}\n"
        f"Vortex Shedding Frequency: {peak_freq:.3f} Hz\n"
        f"Strouhal Number: {strouhal:.4f}\n"
        f"Characteristic Length: {CHARACTERISTIC_LENGTH*1000:.2f} mm\n"
        f"Flow Velocity: {FLOW_VELOCITY:.2f} m/s"
    )
    
    # Place text box
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax4.text(0.98, 0.98, results_text, transform=ax4.transAxes, fontsize=10,
             verticalalignment='top', horizontalalignment='right',
             bbox=props, fontfamily='monospace')
    
    plt.tight_layout()
    return fig


def print_results(peak_freq: float, strouhal: float, fps: float, 
                  total_frames: int, duration: float):
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
    print(f"\nResults:")
    print(f"  • Vortex Shedding Frequency (f): {peak_freq:.3f} Hz")
    print(f"  • Period: {1/peak_freq*1000:.2f} ms")
    print(f"  • Strouhal Number (St = fL/V): {strouhal:.4f}")
    print("\n" + "=" * 60)
    
    # Compare to theoretical values
    print("\nReference: For a circular cylinder, St ≈ 0.2 at Re > 1000")
    if 0.15 < strouhal < 0.25:
        print("Your Strouhal number is in the expected range for cylinder wake!")
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
    
    # Find dominant frequency
    print("Finding dominant frequency...")
    peak_freq, peak_power, peaks = find_dominant_frequency(
        frequencies, power, 
        min_freq=BANDPASS_LOW
    )
    
    # Calculate Strouhal number
    strouhal = calculate_strouhal(peak_freq, CHARACTERISTIC_LENGTH, FLOW_VELOCITY)
    
    # Print results
    print_results(peak_freq, strouhal, fps, total_frames, duration)
    
    # Create visualization
    print("Generating plots...")
    fig = create_visualization(
        roi_frame, time, brightness, processed_signal,
        frequencies, power, peak_freq, strouhal, peaks
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

