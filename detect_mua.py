import numpy as np
from utilities import *

def detect_MUA (data, standard_deviations=5, silent=False):
    # Filter between 0.4-4kHz
    data = butter_bandpass_filter(data, 400, 4000).copy()
    
    # Indices of points below +/- 500
    data_filtered_idxs = np.abs(data) < 500
    
    # Produce threshold as anything below -5 std
    threshold = standard_deviations*np.std(data[data_filtered_idxs])
    
    if not silent:
        print('\tCalculated MUA threshold ({})\n'.format(round(threshold, 2)))
    
    # Find points below threshold
    spike_times = np.where((data < -threshold) & (data > -750))[0]
    
    # Several points in spike waveform might be below threshold
    # So filter out contiguous (t and t+1) timepoints
    spike_times_unique = []
    prev_t = 0
    for t_idx, t in enumerate(spike_times):        
        if prev_t and (prev_t+1 != t):
            spike_times_unique.append(prev_t)
        if (t_idx+1) == len(spike_times) and (prev_t+1 != t):
            spike_times_unique.append(t)
        prev_t = t
    
    # How far to extract waveform in MUA signal from trough
    size = int(SAMPLING_RATE * 1/1000 * 1/2) # 1 ms

    # Extract waveforms and align to trough of waveform
    spike_waveforms = []
    for spike_t in spike_times_unique:
        t_start = np.clip(spike_t - size, a_min=0, a_max=None)
        t_end = np.clip(spike_t + size, a_min=None, a_max=len(data))
        spike_waveform = data[t_start:t_end]
        
        trough_t = np.argmin(spike_waveform) + t_start
        trough_t_start = np.clip(trough_t-size, a_min=0, a_max=None)
        trough_t_end = np.clip(trough_t+size, a_min=None, a_max=len(data))
        
        spike_waveform_aligned = data[trough_t_start:trough_t_end]
        #if np.max(spike_waveform_aligned) > 0 and np.max(spike_waveform_aligned[:15]) > 0:       
        spike_waveforms.append(spike_waveform_aligned)
    
    return spike_times_unique, spike_waveforms

# Get amplitude asymmetry and peak-to-trough width of each spike waveform
# Used to separate excitatory and inhibitory cells
# Based on Reyes-Puerta et al. (2015) (https://academic.oup.com/cercor/article/25/8/2001/310669)
def get_spike_wave_params (swf):
    mid_idx = len(swf)//2

    swf_clipped = np.clip(swf, a_min=0, a_max=None)
    
    peak_a = signal.find_peaks(swf_clipped[:mid_idx])[0]
    if not len(peak_a):
        return False, False
    amp_a = np.max(swf_clipped[:mid_idx][peak_a])

    peak_b = signal.find_peaks(swf_clipped[mid_idx:])[0]
    if not len(peak_b):
        return False, False
    amp_b = np.max(swf_clipped[mid_idx:][peak_b])
    
    amp_asymmetry = (amp_b - amp_a) / (amp_b + amp_a) 
    
    # Trough to peak width
    width = (np.where(swf == amp_b)[0][0] - mid_idx) / SAMPLING_RATE * 1000
    
    return width, amp_asymmetry