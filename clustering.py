import numpy as np

from scipy import signal
from scipy import cluster

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

import skfuzzy as fuzz

from utilities import *
from recording_data import RECORDINGS

####################
# Helper functions #
####################

def rms (data):
    square = [d**2 for d in data]
    mean = np.mean(square)
    root = mean**0.5
    return root
    
def get_feature_vector (burst):
    if not len(burst.data):
        return False
    
    burst_data = butter_bandpass_filter(burst.data, 4, 100)
    
    # Duration
    duration = (burst.time[1] - burst.time[0]) / SAMPLING_RATE
    
    # Spiking activity
    spike_rate = len(burst.spikes) / duration
        
    # Negative peak
    negative_peak = min(burst_data)
    
    # rms
    rms_list = []
    chunk_size = int(SAMPLING_RATE*0.2)
    for i in range(0, len(burst_data), chunk_size):
        data_chunk = burst_data[i:i+chunk_size]
        rms_list.append(rms(data_chunk))
    
    # rms features
    max_rms = np.max(rms_list)
    min_rms = np.min(rms_list)
    flatness = min_rms / max_rms
    
    # slope
    data_downsampled = signal.resample(burst_data, int(len(burst_data)/SAMPLING_RATE * 500))
    max_slope = np.max([x - z for x, z in zip(data_downsampled[:-1], data_downsampled[1:])])
    
    # Beta/low-gamma power
    f_burst, Pxx_burst = burst.normalized_psd
    theta_idx = np.where(f_burst >= 4)[0][0] 
    beta_idx = np.where(f_burst >= 16)[0][0]
    lgamma_idx = np.where(f_burst >= 40)[0][0]
    
    theta_power = np.sum(Pxx_burst[theta_idx:beta_idx]) / np.sum(Pxx_burst[theta_idx:])
    
    beta_lgamma_power = np.sum(Pxx_burst[beta_idx:lgamma_idx]) / np.sum(Pxx_burst[theta_idx:])
    
    # Peak/trough features
    peaks, cycles = get_peaks(burst_data)
    iti = np.mean(np.diff([p for p in peaks if burst_data[p] < 0])) / SAMPLING_RATE
    if np.isnan(iti):
        return False
        
    return [
        duration,
        max_rms,
        negative_peak,
        flatness,
        max_slope,
        beta_lgamma_power,
        theta_power,
        iti,
        spike_rate
    ]

feature_vector_labels = [
    "Duration",
    "Max RMS",
    "Negative peak",
    "Flatness",
    "Max slope",
    "β-γ power",
    "θ-α power",
    "ITI",
    "Spike rate"
]
feature_vector_labels_full = [
    "Duration (s)",
    "Max RMS (μV)",
    "Negative peak (μV)",
    "Flatness",
    "Max slope",
    "Beta/low-gamma power",
    "Theta-alpha power",
    "Inter-trough-interval (s)",
    "Spikes $s^{-1}$"
]

####################
# Get feature vecs #
####################

def get_feature_vectors (rms_processed_recordings, processed_recordings_mua, key, mua_key):
    all_bursts = []
    all_features = []

    print('Processing bursts to feature vectors')
    for idx, recording in enumerate(rms_processed_recordings):
        print("\t{} (P{}) {}/{}".format(recording["path"], recording["age"], idx+1, len(rms_processed_recordings)))

        if not key in recording:
            continue

        spike_times_exist = False
        for r in processed_recordings_mua:
            if r['path'] == recording['path'] and r['recording'] == recording['recording']:
                spike_times = r[mua_key][0]
                spike_times_exist = True
        if not spike_times_exist:
            continue
                
        for burst_idx, burst in enumerate(recording[key]):
            if burst.primary_frequency_baseline:
                if not hasattr(burst, 'spikes'):
                    # Spikes in range of burst
                    spikes_in_range = [t for t in spike_times if (t >= burst.time[0] and t <= burst.time[1])]
                    burst.spikes = spikes_in_range
                
                if not hasattr(burst, 'feature_vec'):
                    burst.feature_vec = get_feature_vector(burst)        
                        
                if hasattr(burst, 'feature_vec') and burst.feature_vec and burst.feature_vec[0] < 20:
                    burst.age = recording["age"]
                    burst.recording_path = recording["path"]

                    all_features.append(burst.feature_vec)
                    all_bursts.append(burst)
        
    all_bursts = np.array(all_bursts)
    all_features = np.array(all_features)
        
    return all_bursts, all_features

##############
# Clustering #
##############
                  
# Fuzzy cluster with thresh
def fuzzy_cluster (samples, n_clusters, threshold=0.6):
    cntr, u, u0, d, jm, p, fpc = fuzz.cmeans(samples.T, n_clusters, 2, error=0.005, maxiter=10000, init=None)
    
    labels = []
    for sample in u.T:
        max_p = np.max(sample)
        if max_p > threshold:
            labels.append(np.argmax(sample, axis=0))
        else:
            labels.append(len(sample))
    labels = np.array(labels)
    
    return labels, fpc
    
def get_cluster_labels (all_features, pca=None):
    # Run PCA analysis using first 3 components
    if pca:
        scaler = StandardScaler()
        pc_feature_list = pca.transform(scaler.fit_transform(all_features))
    else:
        pca = PCA(n_components=2)
        scaler = StandardScaler()
        pc_feature_list = pca.fit_transform(scaler.fit_transform(all_features))
        
    # Get optimal number of clusters
    n_clusters = np.arange(2, 6)
    silhouette_scores = []

    for n in n_clusters:
        _, fpc = fuzzy_cluster(pc_feature_list, n)
        silhouette_scores.append(fpc)

    optimal_n_clusters = n_clusters[np.argmax(silhouette_scores)]
    optimal_n_clusters = 2
    
    # Plot clusters (of first 3 components)
    cluster_data = {
        "cluster": [],
        "pc1": [],
        "pc2": [],
        "pc3": []
    }

    labels, _ = fuzzy_cluster(pc_feature_list, optimal_n_clusters)
    
    return labels, pca

def sort_bursts_by_labels (all_bursts, all_features, all_labels):    
    gamma_0 = np.mean(all_features[all_labels==0][:, 5])
    gamma_1 = np.mean(all_features[all_labels==1][:, 5])
    
    if gamma_0 > gamma_1:
        NGB_LABEL = 0
        SB_LABEL = 1
    else:
        NGB_LABEL = 1
        SB_LABEL = 0
        
    feature_list_ngb = all_features[all_labels == NGB_LABEL]
    feature_list_sb = all_features[all_labels == SB_LABEL]

    bursts_ngb = all_bursts[all_labels == NGB_LABEL]
    bursts_sb = all_bursts[all_labels == SB_LABEL]
    
    return feature_list_ngb, feature_list_sb, bursts_ngb, bursts_sb
