"""
Multi-ROI Segmented Vortex Shedding Analyzer (4 ROIs + 5 Segments)
==================================================================
Analyzes wind tunnel video by splitting it into temporal segments and
tracking brightness in 4 spatial ROIs. Computes advanced signal quality
metrics and error analysis.

Author: MAE 157W Lab 1 (Refactored for Segmentation & Metrics)
"""

import cv2
import numpy as np
import pandas as pd
from scipy import signal
from scipy.fft import rfft, rfftfreq
import matplotlib.pyplot as plt
from pathlib import Path

# =============================================================================
# CONFIGURATION
# =============================================================================

VIDEO_PATH = "Laser1.mov"

# --- ROIs (x, y, w, h) ---
ROIS = [
    {"coords": (775, 550, 62, 62), "label": "ROI 1 (0.5-1.0D, Narrow)"},
    {"coords": (775, 550, 125, 62), "label": "ROI 2 (0.5-1.0D, Wide)"},
    {"coords": (900, 550, 62, 62), "label": "ROI 3 (1.0-1.5D, Narrow)"},
    {"coords": (900, 550, 125, 62), "label": "ROI 4 (1.0-1.5D, Wide)"}
]

# --- Segmentation Settings ---
NUM_SEGMENTS = 5  # Split video into this many equal time chunks

# --- Physical Parameters & Uncertainty ---
# St = f * L / V
L_CHAR = 0.059       # meters (Diameter)
L_UNCERTAINTY = 0.001 # meters (e.g. 0.5mm uncertainty)

V_FLOW = 0.363       # m/s
V_UNCERTAINTY = 0.0364 # m/s (e.g. 0.01 m/s uncertainty)

# --- Signal Processing ---
FRAME_RATE = 240         # Set None to auto-detect
BANDPASS_LOW = 0.5       
BANDPASS_HIGH = 5.0      
USE_BANDPASS = True      
NORMALIZE_FRAMES = True  
DETREND_BRIGHTNESS = True

# =============================================================================
# DATA STRUCTURES
# =============================================================================

class VideoLoader:
    @staticmethod
    def load_info(path):
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open {path}")
        fps = cap.get(cv2.CAP_PROP_FPS)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        return cap, fps, total

    @staticmethod
    def extract_all_brightness(cap, rois, total_frames, normalize=True):
        """Extracts full time series for all ROIs at once. Also captures first frame with ROIs."""
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        data = [[] for _ in rois]
        frame_means = []
        roi_frame = None  # Store first frame with ROI overlay
        
        # Define colors for each ROI (BGR format)
        colors = [(0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 0, 255)]  # Green, Blue, Yellow, Magenta
        
        print(f"Extracting full video brightness ({total_frames} frames)...")
        for i in range(total_frames):
            ret, frame = cap.read()
            if not ret: break
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if normalize:
                frame_means.append(np.mean(gray))
            
            # Store first frame with ROI overlays
            if i == 0:
                roi_frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                for idx, roi in enumerate(rois):
                    x, y, w, h = roi["coords"]
                    color = colors[idx % len(colors)]
                    cv2.rectangle(roi_frame, (x, y), (x+w, y+h), color, 2)
                    cv2.putText(roi_frame, f"ROI {idx+1}", (x, y-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
            for idx, roi in enumerate(rois):
                x, y, w, h = roi["coords"]
                data[idx].append(np.mean(gray[y:y+h, x:x+w]))
                
            if (i+1) % 500 == 0:
                print(f"  Frame {i+1}/{total_frames}")

        # Post-process arrays
        processed_data = []
        raw_data = []
        frame_means = np.array(frame_means)
        
        for raw_list in data:
            raw = np.array(raw_list)
            raw_data.append(raw.copy())
            # Normalize
            if normalize and len(frame_means) > 0:
                sig = (raw / frame_means) * np.mean(frame_means)
            else:
                sig = raw
            
            processed_data.append(sig)
            
        return raw_data, processed_data, roi_frame

# =============================================================================
# ANALYSIS LOGIC
# =============================================================================

def calculate_uncertainty(f, L, V, dL, dV, df=0.0):
    """
    Calculates standard deviation of Strouhal number based on error propagation.
    St = f * L / V
    (dSt/St)^2 = (df/f)^2 + (dL/L)^2 + (dV/V)^2
    Assuming df (freq error) is negligible compared to physical measurement error usually, 
    but can be added if resolution is known.
    """
    St = (f * L) / V
    if f == 0: return 0.0, 0.0
    
    # Relative variances
    rel_var_L = (dL / L) ** 2
    rel_var_V = (dV / V) ** 2
    rel_var_f = (df / f) ** 2
    
    rel_uncertainty = np.sqrt(rel_var_f + rel_var_L + rel_var_V)
    abs_uncertainty = St * rel_uncertainty
    
    return St, abs_uncertainty

def analyze_segment(signal_seg, fps):
    """
    Computes metrics for a single segment of data.
    """
    n = len(signal_seg)
    if n < 2: return None
    
    # 1. Preprocess (Detrend & Filter)
    sig_dt = signal.detrend(signal_seg, type='linear')
    if USE_BANDPASS and fps > 2*BANDPASS_HIGH:
        nyq = 0.5 * fps
        b, a = signal.butter(4, [BANDPASS_LOW/nyq, BANDPASS_HIGH/nyq], btype='band')
        filtered = signal.filtfilt(b, a, sig_dt)
    else:
        filtered = sig_dt

    # 2. FFT Analysis & FQ (FFT Peak Quality)
    window = np.hanning(n)
    fft_res = rfft(filtered * window)
    freqs = rfftfreq(n, d=1/fps)
    power = np.abs(fft_res) ** 2
    
    # Find peak
    valid_mask = (freqs >= BANDPASS_LOW) & (freqs <= BANDPASS_HIGH)
    if not np.any(valid_mask):
        f_fft = 0.0
        FQ = 0.0
    else:
        # Restrict search to valid band
        idx_search = np.where(valid_mask)[0]
        idx_peak_local = np.argmax(power[valid_mask])
        idx_peak = idx_search[idx_peak_local]
        f_fft = freqs[idx_peak]
        amp_peak = power[idx_peak]
        
        # FQ Calculation: Peak vs Local Noise
        # Define noise band: +/- 1 Hz around peak, excluding +/- 0.1 Hz around peak
        mask_noise = (freqs > f_fft - 1.0) & (freqs < f_fft + 1.0)
        mask_peak_excl = (freqs > f_fft - 0.2) & (freqs < f_fft + 0.2)
        mask_final = mask_noise & (~mask_peak_excl)
        
        if np.any(mask_final):
            noise_floor = np.mean(power[mask_final])
            FQ = amp_peak / noise_floor if noise_floor > 0 else 0.0
        else:
            FQ = 0.0

    # 3. Autocorrelation & PS/PSRQ
    norm_sig = (filtered - np.mean(filtered)) / (np.std(filtered) + 1e-9)
    ac_full = np.correlate(norm_sig, norm_sig, mode='full')
    ac = ac_full[n-1:]
    ac = ac / ac[0] # Normalize R(0)=1
    lags = np.arange(len(ac)) / fps
    
    # Find peaks for PS
    # Min distance = 1/max_freq
    min_dist = int(fps / BANDPASS_HIGH)
    peaks, _ = signal.find_peaks(ac, height=0.0, distance=min_dist)
    
    # Ignore peaks at very low lag (noise)
    peaks = [p for p in peaks if lags[p] > 0.5/BANDPASS_HIGH]
    
    if len(peaks) > 0:
        first_peak_idx = peaks[0]
        T_period = lags[first_peak_idx]
        f_ac = 1.0 / T_period
        
        # B) PS: Strength of first peak
        PS = ac[first_peak_idx] # Already normalized by R(0)
        
        # C) PSRQ: Peak to Background Std Dev
        # Background is everything not near 0 and not near the peak? 
        # Simpler definition: Std dev of AC tail
        mask_bg = (lags > T_period * 1.5)
        if np.any(mask_bg):
            bg_std = np.std(ac[mask_bg])
            PSRQ = PS / bg_std if bg_std > 0 else 0.0
        else:
            PSRQ = 0.0
    else:
        f_ac = 0.0
        PS = 0.0
        PSRQ = 0.0

    # D) Method Agreement (FA)
    if f_fft > 0 and f_ac > 0:
        FA = abs(f_fft - f_ac) / ((f_fft + f_ac)/2) * 100
    else:
        FA = np.nan

    return {
        "f_fft": f_fft,
        "f_ac": f_ac,
        "FQ": FQ,
        "PS": PS,
        "PSRQ": PSRQ,
        "FA": FA
    }

def compile_roi_statistics(roi_label, segment_results):
    """
    Aggregates results from multiple segments for a single ROI.
    """
    df = pd.DataFrame(segment_results)
    
    # Averages
    mean_fft = df["f_fft"].mean()
    mean_ac = df["f_ac"].mean()
    mean_FQ = df["FQ"].mean()
    mean_PS = df["PS"].mean()
    mean_PSRQ = df["PSRQ"].mean()
    mean_FA = df["FA"].mean()
    
    # E) Repeatability (RepCV)
    # Using FFT freq for consistency check
    if mean_fft > 0:
        rep_cv_fft = (df["f_fft"].std() / mean_fft) * 100
    else:
        rep_cv_fft = 0.0
    
    if mean_ac > 0:
        rep_cv_ac = (df["f_ac"].std() / mean_ac) * 100
    else:
        rep_cv_ac = 0.0
        
    # F) Strouhal Error - Calculate for both FFT and AC frequencies
    st_fft, st_fft_err = calculate_uncertainty(mean_fft, L_CHAR, V_FLOW, 
                                               L_UNCERTAINTY, V_UNCERTAINTY)
    st_ac, st_ac_err = calculate_uncertainty(mean_ac, L_CHAR, V_FLOW, 
                                             L_UNCERTAINTY, V_UNCERTAINTY)
    
    return {
        "ROI": roi_label,
        "f_fft": mean_fft,
        "f_ac": mean_ac,
        "FQ": mean_FQ,
        "PS": mean_PS,
        "PSRQ": mean_PSRQ,
        "FA_pct": mean_FA,
        "RepCV_fft": rep_cv_fft,
        "RepCV_ac": rep_cv_ac,
        "St_fft": st_fft,
        "St_fft_err": st_fft_err,
        "St_ac": st_ac,
        "St_ac_err": st_ac_err
    }


# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def create_signal_figure(roi_frame, time, raw_signals, processed_signals, rois, fps):
    """
    Create Figure 1: Video frame with ROIs and extracted/processed signals.
    """
    n_rois = len(rois)
    colors = ['green', 'blue', 'orange', 'magenta']
    
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle('Vortex Shedding Analysis - Signal Extraction', fontsize=14, fontweight='bold')
    
    # Layout: 3 rows - Frame | Raw Signals | Processed Signals
    gs = fig.add_gridspec(3, 1, height_ratios=[1.2, 1, 1], hspace=0.35)
    
    # Panel 1: Video frame with ROIs
    ax1 = fig.add_subplot(gs[0])
    if roi_frame is not None:
        ax1.imshow(cv2.cvtColor(roi_frame, cv2.COLOR_BGR2RGB))
    ax1.set_title('Video Frame with ROI Locations', fontweight='bold')
    ax1.set_xlabel('X (pixels)')
    ax1.set_ylabel('Y (pixels)')
    
    # Panel 2: Raw brightness signals
    ax2 = fig.add_subplot(gs[1])
    for i, (raw, roi) in enumerate(zip(raw_signals, rois)):
        ax2.plot(time, raw, color=colors[i % len(colors)], linewidth=0.5, 
                 alpha=0.8, label=roi["label"])
    ax2.set_title('Raw Brightness Signals', fontweight='bold')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Mean Brightness')
    ax2.legend(loc='upper right', fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    # Panel 3: Processed (normalized) signals
    ax3 = fig.add_subplot(gs[2])
    for i, (proc, roi) in enumerate(zip(processed_signals, rois)):
        # Apply detrending for display
        proc_dt = signal.detrend(proc, type='linear')
        ax3.plot(time, proc_dt, color=colors[i % len(colors)], linewidth=0.5, 
                 alpha=0.8, label=roi["label"])
    ax3.set_title('Processed Signals (Normalized & Detrended)', fontweight='bold')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Brightness Deviation')
    ax3.legend(loc='upper right', fontsize=8)
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def create_analysis_figure(full_signals, rois, fps, total_frames):
    """
    Create Figure 2: FFT Power Spectrum and Autocorrelation for each ROI.
    Uses the first segment for display.
    """
    n_rois = len(rois)
    colors = ['green', 'blue', 'orange', 'magenta']
    
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle('Frequency Analysis - FFT & Autocorrelation Comparison', 
                 fontsize=14, fontweight='bold')
    
    # Layout: 2 rows (FFT, Autocorr) x n_rois columns
    gs = fig.add_gridspec(2, n_rois, hspace=0.35, wspace=0.3)
    
    frames_per_seg = total_frames // NUM_SEGMENTS
    
    for r_idx, (sig_full, roi) in enumerate(zip(full_signals, rois)):
        # Use first segment for visualization
        segment = sig_full[:frames_per_seg]
        n = len(segment)
        
        # Preprocess
        sig_dt = signal.detrend(segment, type='linear')
        if USE_BANDPASS and fps > 2*BANDPASS_HIGH:
            nyq = 0.5 * fps
            b, a = signal.butter(4, [BANDPASS_LOW/nyq, BANDPASS_HIGH/nyq], btype='band')
            filtered = signal.filtfilt(b, a, sig_dt)
        else:
            filtered = sig_dt
        
        # FFT
        window = np.hanning(n)
        fft_res = rfft(filtered * window)
        freqs = rfftfreq(n, d=1/fps)
        power = np.abs(fft_res) ** 2
        power_norm = power / np.max(power) if np.max(power) > 0 else power
        
        # Find FFT peak
        valid_mask = (freqs >= BANDPASS_LOW) & (freqs <= BANDPASS_HIGH)
        if np.any(valid_mask):
            idx_search = np.where(valid_mask)[0]
            idx_peak = idx_search[np.argmax(power[valid_mask])]
            f_fft = freqs[idx_peak]
        else:
            f_fft = 0
        
        # Autocorrelation
        norm_sig = (filtered - np.mean(filtered)) / (np.std(filtered) + 1e-9)
        ac_full = np.correlate(norm_sig, norm_sig, mode='full')
        ac = ac_full[n-1:]
        ac = ac / ac[0]
        lags = np.arange(len(ac)) / fps
        
        # Find AC peaks
        min_dist = int(fps / BANDPASS_HIGH)
        peaks, _ = signal.find_peaks(ac, height=0.0, distance=min_dist)
        peaks = [p for p in peaks if lags[p] > 0.5/BANDPASS_HIGH]
        
        if len(peaks) > 0:
            f_ac = 1.0 / lags[peaks[0]]
        else:
            f_ac = 0
        
        # Plot FFT (top row)
        ax_fft = fig.add_subplot(gs[0, r_idx])
        ax_fft.semilogy(freqs, power_norm, color=colors[r_idx % len(colors)], linewidth=1)
        ax_fft.axvline(x=f_fft, color='red', linestyle='--', linewidth=1.5, 
                       label=f'Peak: {f_fft:.2f} Hz')
        ax_fft.set_title(f'{roi["label"]}\nFFT Spectrum', fontweight='bold', fontsize=10)
        ax_fft.set_xlabel('Frequency (Hz)')
        ax_fft.set_ylabel('Norm. Power')
        ax_fft.set_xlim([0, BANDPASS_HIGH * 1.5])
        ax_fft.legend(loc='upper right', fontsize=8)
        ax_fft.grid(True, alpha=0.3)
        
        # Plot Autocorrelation (bottom row)
        ax_ac = fig.add_subplot(gs[1, r_idx])
        max_lag_display = min(3.0, lags[-1])  # Show up to 3 seconds
        mask = lags <= max_lag_display
        ax_ac.plot(lags[mask], ac[mask], color=colors[r_idx % len(colors)], linewidth=1)
        ax_ac.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
        
        # Mark peaks
        valid_peaks = [p for p in peaks if p < len(lags[mask])]
        if len(valid_peaks) > 0:
            ax_ac.plot(lags[valid_peaks], ac[valid_peaks], 'r^', markersize=6)
            if f_ac > 0:
                ax_ac.axvline(x=1/f_ac, color='red', linestyle='--', linewidth=1.5,
                             label=f'T={1000/f_ac:.1f}ms ({f_ac:.2f}Hz)')
        
        ax_ac.set_title(f'Autocorrelation', fontweight='bold', fontsize=10)
        ax_ac.set_xlabel('Lag (s)')
        ax_ac.set_ylabel('Correlation')
        ax_ac.set_xlim([0, max_lag_display])
        ax_ac.set_ylim([-0.5, 1.1])
        ax_ac.legend(loc='upper right', fontsize=8)
        ax_ac.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

# =============================================================================
# MAIN PIPELINE
# =============================================================================

def main():
    print("="*80)
    print(f"MULTI-SEGMENT VORTEX ANALYZER ({len(ROIS)} ROIs, {NUM_SEGMENTS} Segments)")
    print("="*80)
    
    # 1. Load Video & Extract Full Signals
    if not Path(VIDEO_PATH).exists():
        print(f"Error: {VIDEO_PATH} not found.")
        return

    cap, fps_vid, total_frames = VideoLoader.load_info(VIDEO_PATH)
    fps = FRAME_RATE if FRAME_RATE else fps_vid
    
    # Extract continuous signals for all ROIs (now returns raw, processed, and frame)
    raw_signals, full_signals, roi_frame = VideoLoader.extract_all_brightness(
        cap, ROIS, total_frames, NORMALIZE_FRAMES
    )
    cap.release()
    
    # Use actual extracted frames (may be less than total_frames if some failed to read)
    actual_frames = len(raw_signals[0])
    
    # Create time array based on actual extracted frames
    time_arr = np.arange(actual_frames) / fps

    # 2. Segment Analysis
    frames_per_seg = actual_frames // NUM_SEGMENTS
    print(f"\nSegmentation: {NUM_SEGMENTS} segments of ~{frames_per_seg/fps:.2f}s each.")
    
    final_report_rows = []

    for r_idx, roi_conf in enumerate(ROIS):
        roi_name = roi_conf["label"]
        raw_signal = full_signals[r_idx]
        
        # Store results for each segment of this ROI
        seg_results = []
        
        print(f"  Analzying {roi_name}...")
        for s in range(NUM_SEGMENTS):
            start = s * frames_per_seg
            end = start + frames_per_seg
            
            # Slice the signal
            segment_data = raw_signal[start:end]
            
            # Run analysis
            res = analyze_segment(segment_data, fps)
            if res:
                seg_results.append(res)
        
        # Compile statistics for this ROI across all segments
        roi_stats = compile_roi_statistics(roi_name, seg_results)
        final_report_rows.append(roi_stats)

    # 3. Final Report Table
    print("\n" + "="*100)
    print("FINAL COMPILED METRICS TABLE")
    print("="*100)
    
    # Define columns for display
    # FQ=FFT Quality, PS=Autocorr Strength, PSRQ=Peak/Backg, FA=Method Diff, RepCV=Variation
    headers = [
        "ROI", "f_fft (Hz)", "St_fft ± Err", "f_ac (Hz)", "St_ac ± Err",
        "FQ", "PS", "PSRQ", "FA (%)", "RepCV (%)"
    ]
    
    # Print Header
    header_fmt = "{:<22} | {:<10} | {:<14} | {:<10} | {:<14} | {:<6} | {:<6} | {:<6} | {:<8} | {:<9}"
    print(header_fmt.format(*headers))
    print("-" * 135)
    
    for row in final_report_rows:
        st_fft_str = f"{row['St_fft']:.3f}±{row['St_fft_err']:.3f}"
        st_ac_str = f"{row['St_ac']:.3f}±{row['St_ac_err']:.3f}"
        
        print(header_fmt.format(
            row["ROI"],
            f"{row['f_fft']:.3f}",
            st_fft_str,
            f"{row['f_ac']:.3f}",
            st_ac_str,
            f"{row['FQ']:.1f}",
            f"{row['PS']:.3f}",
            f"{row['PSRQ']:.1f}",
            f"{row['FA_pct']:.1f}",
            f"{row['RepCV_fft']:.1f}"
        ))
    print("="*135)
    print("Legend:")
    print("  St_fft: Strouhal number from FFT frequency")
    print("  St_ac:  Strouhal number from Autocorrelation frequency")
    print("  FQ:     FFT Peak Quality (Peak Amp / Local Noise)")
    print("  PS:     Periodicity Strength (Autocorr Peak Height)")
    print("  PSRQ:   Peak-to-Background Ratio (Autocorr Peak / Background Std)")
    print("  FA:     Frequency Agreement % between FFT and Autocorr")
    print("  RepCV:  Coeff of Variation of FFT frequency across segments (Repeatability)")
    print("="*135)
    
    # 4. Generate Visualization Figures
    print("\nGenerating visualization figures...")
    
    # Figure 1: Video frame with ROIs and signal plots
    fig_signal = create_signal_figure(roi_frame, time_arr, raw_signals, full_signals, ROIS, fps)
    
    # Figure 2: FFT and Autocorrelation analysis
    fig_analysis = create_analysis_figure(full_signals, ROIS, fps, actual_frames)
    
    # Save figures
    base_name = Path(VIDEO_PATH).stem
    
    signal_path = base_name + "_signal.png"
    fig_signal.savefig(signal_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {signal_path}")
    
    analysis_path = base_name + "_analysis.png"
    fig_analysis.savefig(analysis_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {analysis_path}")
    
    # Show figures
    plt.show()
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()