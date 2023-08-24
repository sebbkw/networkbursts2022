import pickle
import os
from datetime import datetime

import pickle5

import numpy as np
import pandas as pd
from spectrum import dpss, pmtm

import scipy
from scipy import signal
from scipy import io
from scipy import ndimage
from scipy import stats

from statsmodels.stats.descriptivestats import sign_test
import statsmodels.api as sm
from statsmodels.formula.api import ols

from matplotlib import pyplot as plt
plt.rcParams["font.family"] = "Helvetica"


###########
# Classes #
###########

class Burst:
    def __init__ (self, time, data):
        self.time = time
        self.data = data
    
    def get_primary_frequency (self, exclude_electrical_noise=True):
        f, Pxx = multitaper_psd(self.data)     
        
        fm = FOOOF(verbose=False)
        fm.fit(f, Pxx, freq_range=[1, 80])
        
        peaks = fm.get_results().peak_params
        peaks = [peak[0] for peak in sorted(peaks, key=lambda val: val[1], reverse=True)]
        if exclude_electrical_noise:
            peaks = [peak for peak in peaks if (peak > 54 or peak < 49)]
        
        if len(peaks):
            self.primary_frequency, self.r_squared = peaks[0], fm.r_squared_
            
            return peaks[0], fm.r_squared_
        else:
            self.primary_frequency, self.r_squared = False, False
            
            return False, False
        
    def get_xticks (self):
        return get_xticks(slice(*self.time))

############################
# Graph plotting functions #
############################
def format_plot (ax, legend=True, size=[12.5, 15]): 
    if legend:
        leg = plt.legend(frameon=False, bbox_to_anchor=(1, 1), loc='upper left')
        for legobj in leg.legendHandles:
            legobj.set_linewidth(2.0)
            legobj.set_alpha(1)
        for i in ax.get_legend().get_texts():
            i.set_fontsize(size[1])
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    for item in ([ax.title] + ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(size[0])
    for item in ([ax.xaxis.label, ax.yaxis.label]):
        item.set_fontsize(size[1])

def plot_graph_by_age (data, ylabel, scatter, colors, labels, save=False, log=False):
    for brain_idx, brain_area in enumerate(data.keys()):
        data_by_age = data[brain_area]
        xpos = np.arange(len(data_by_age.keys()))
        xlabels = [age for age in data_by_age.keys()]
        
        mean = [np.mean(d) for d in data_by_age.values()]
        stderr = [np.std(d)/len(d)**0.5 for d in data_by_age.values()]

        if scatter:
            xpos_scatter, value_scatter = [], []
            for idx, d in enumerate(data_by_age.values()):
                for val in d:
                    xpos_scatter.append(idx)
                    value_scatter.append(val)
            plt.plot(xpos_scatter, value_scatter, c=colors[brain_idx], lw=0, marker='o', alpha=0.25)

        plt.errorbar(xpos, mean, yerr=stderr, capsize=5, c=colors[brain_idx], label=labels[brain_idx])
        plt.xticks(xpos, xlabels)
        plt.xlabel('Age (days)')
        plt.ylabel(ylabel)
        if log:
            plt.yscale('log')
        
    ax = plt.gca()
        
    handles, labels = ax.get_legend_handles_labels()
    handles = [h[0] for h in handles]
    leg = ax.legend(handles, labels, frameon=False, bbox_to_anchor=(1, 1), loc='upper left')

    for legobj in leg.legendHandles:
        legobj.set_linewidth(2.0)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    for item in ([ax.title] + ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(12.5)
    for item in ([ax.xaxis.label, ax.yaxis.label] + ax.get_legend().get_texts()):
        item.set_fontsize(15)
    
    if save:
        plt.savefig(FIG_ROOT + save, bbox_inches="tight")
    plt.show()
    
def plot_threshold (ax, data_range, threshold):
    x = get_xticks(data_range)
    y = np.ones(len(x)) * threshold
    
    ax.plot(x, y)
    ax.plot(x, -y)
    
def plot_frequency_bands (ax):
    ax.axvline(x=4, ymin=0, ymax=1, c='orange') # Alpha
    ax.axvline(x=8, ymin=0, ymax=1, c='orange') # Theta
    ax.axvline(x=13, ymin=0, ymax=1, c='orange') # Beta
    ax.axvline(x=30, ymin=0, ymax=1, c='orange') # Gamma
    ax.axvline(x=80, ymin=0, ymax=1, c='orange')
    
def plot_spectrogram (burst, ax=None):    
    freq_lim = 50
    window_size = 0.5
    
    if len(burst.data)/SAMPLING_RATE <= 0.2:
        window_size = 0.1
    
    window = int(SAMPLING_RATE*window_size)
    overlap = int(SAMPLING_RATE*window_size*0.99)
    
    f, t, Sxx = signal.spectrogram(burst.data, SAMPLING_RATE, nperseg=window, noverlap=overlap)
    
    f_lim = np.where(f <= freq_lim+25)[0][-1]
    f = f[:f_lim]
    Sxx = Sxx[:f_lim, :]
    
    t += burst.time[0]/SAMPLING_RATE
    
    if ax == None:
        fig, ax = plt.subplots()
    
    ax.pcolormesh(t, f, ndimage.gaussian_filter(Sxx, sigma=0), shading='gouraud', vmax=np.percentile(Sxx.flatten(), 98))
    ax.set_ylim([0, freq_lim])
    ax.set_yticks(np.arange(0, freq_lim, 20))    
    
def plot_psd (data):
    f, Pxx = multitaper_psd(data)
    
    # Find 0-100 Hz range
    max_idx = np.where(f <= 100)[0][-1]
    f = f[:max_idx]
    Pxx = Pxx[:max_idx]
    
    # Scale total range to 1
    #Pxx = scale_0_to_1 (Pxx)

    fig, ax = plt.subplots(figsize=[10,2])
    plt.plot(f, Pxx)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power spectral density')
    plt.xlim([0, 100])
    plot_frequency_bands(ax)
    plt.show()
    
def plot_group_boxplots (data, labels, xticks, colors=['blue', 'orange'], title='', xlabel='', ylabel='', yscale='linear', save=False):
    def set_box_color(bp, color):
        plt.setp(bp['boxes'], color=color)
        plt.setp(bp['whiskers'], color=color)
        plt.setp(bp['caps'], color=color)
        plt.setp(bp['medians'], color=color)
    
    fig = plt.figure()
    
    boxplots = []
    for idx, data_group in enumerate(data):
        shift = None
        
        if len(data) == 1:
            shift = 0
        elif len(data) == 2 and idx == 0:
            shift = -0.4
        elif len(data) ==2 and idx == 1:
            shift = 0.4

        positions = np.array(range(len(data_group))) * 2 + shift
        flier = dict(marker='o', markerfacecolor='none', markeredgecolor=colors[idx], alpha=0.5)
        bp = plt.boxplot(data_group, positions=positions, widths=0.6, flierprops=flier)
        set_box_color(bp, colors[idx])
        
        boxplots.append(bp)
    plt.xticks(range(0, len(xticks) * 2, 2), xticks)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.yscale(yscale)
        
    leg = plt.legend([bp["boxes"][0] for bp in boxplots], labels, frameon=False, fontsize=15, bbox_to_anchor=(1, 1), loc='upper left')
    for legobj in leg.legendHandles:
        legobj.set_linewidth(2.0)
        
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    for item in ([ax.title] + ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(12.5)
    for item in ([ax.xaxis.label, ax.yaxis.label] + ax.get_legend().get_texts()):
        item.set_fontsize(15)
    
    if save:
        plt.savefig(FIG_ROOT + save, bbox_inches="tight")
    plt.show()

#####################
# Utility functions #
#####################

def get_spikes_in_range (data_range, spike_times):
    spike_times = np.array(spike_times)
    
    start = data_range.start
    stop = data_range.stop  
    spike_time_idxs = np.where(np.logical_and(spike_times >= start, spike_times <= stop))[0]    
    
    return spike_times[spike_time_idxs]

def get_slice_from_s (s_beg, s_end):
    return slice(int(s_beg*SAMPLING_RATE), int(s_end*SAMPLING_RATE))

def get_xticks (slice_val):
    start, stop = slice_val.start, slice_val.stop
    return np.linspace(start, stop, num=stop-start) / SAMPLING_RATE

def save_data (data, filename=None):
    curr_time = datetime.today().strftime('%Y-%m-%d-%H-%M-%S')    
    path =  filename + '_' + curr_time + '.pickle'
    
    with open(path, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    print('Saved data as', path)
    
def open_data (filename):
    path = os.path.join(filename)
    
    with open(path, 'rb') as handle:
        try:
            data = pickle.load(handle)
        except:
            data = pickle5.load(handle)
    return data
    
#########################################
# Signal processing and stats functions #
#########################################

def welch_dof(x,y):
    dof = (np.var(x)/len(x) + np.var(y)/len(y))**2 / ((np.var(x)/len(x))**2 / (len(x)-1) + (np.var(y)/len(y))**2 / (len(y)-1))
    return dof 

def get_peaks (data):
    min_prominence = 0.5*rms(data)
    min_lag = int(SAMPLING_RATE * 0.025)
    
    peaks = signal.find_peaks(data, distance=min_lag, prominence=min_prominence)[0]
    troughs = signal.find_peaks(-data, distance=min_lag, prominence=min_prominence)[0]

    combined = np.array(sorted(np.concatenate([peaks, troughs])))

    peak_ranks = []
    trough_ranks = []

    for point_idx, point in enumerate(combined):
        if point in peaks:
            peak_ranks.append(point_idx)
        else:
            trough_ranks.append(point_idx)  
     
    def get_max_height_from_ranks (ranks):
        heights = [abs(data[combined[rank]]) for rank in ranks]
        
        return ranks[np.argmax(heights)]
    
    def get_nonadjacent_ranks (ranks):
        nonadjacent_ranks = []
        temp_ranks = []
        for rank_idx, curr_rank in enumerate(ranks):
            if rank_idx == 0:
                temp_ranks.append(curr_rank)
            else:
                prev_rank = ranks[rank_idx-1]

                if prev_rank+1 == curr_rank:
                    temp_ranks.append(curr_rank)
                else:
                    nonadjacent_ranks.append(get_max_height_from_ranks(temp_ranks))
                    temp_ranks = [curr_rank]

            if rank_idx+1 == len(ranks):
                nonadjacent_ranks.append(get_max_height_from_ranks(temp_ranks))

        return nonadjacent_ranks
        
    filtered_ranks = sorted(
        get_nonadjacent_ranks (peak_ranks) + 
        get_nonadjacent_ranks (trough_ranks)
    )
    filtered_points = combined[filtered_ranks]

    return filtered_points, len(get_nonadjacent_ranks (peak_ranks))

def twoway_anova (data, variable_labels):
    factor_1_var, factor_2_var, value_var = variable_labels
    
    prepared_data = {}
    prepared_data[factor_1_var] = []
    prepared_data[factor_2_var] = []
    prepared_data[value_var] = []
    
    for factor_1 in data.keys():
        factor_2_values = data[factor_1]
        for factor_2 in factor_2_values.keys():
            values = factor_2_values[factor_2]
            for value in values:
                prepared_data[factor_1_var].append(factor_1) 
                prepared_data[factor_2_var].append(factor_2) 
                prepared_data[value_var].append(value) 
    
    df = pd.DataFrame(prepared_data)
    
    formula = "{value_var} ~ C({factor_1_var}) + C({factor_2_var}) + C({factor_1_var}):C({factor_2_var})".format(
        factor_1_var=factor_1_var,
        factor_2_var=factor_2_var,
        value_var=value_var
    )
    model = ols(formula, data=df).fit()
    
    return sm.stats.anova_lm(model, typ=2)

def get_mean_psd (bursts):
    fs = None
    Pxxs = []
    
    for burst in bursts:
        if hasattr(burst, "normalized_psd"):
            f_burst, Pxx_burst = burst.normalized_psd
            
            fs = f_burst
            Pxxs.append(Pxx_burst)

    Pxx_mean = np.mean(Pxxs, axis=0)
    Pxx_stderr = np.std(Pxxs, axis=0) / len(Pxxs)**0.5
    
    f, Pxx_mean = get_psd_in_range((fs, Pxx_mean), [0, 40])
    f, Pxx_stderr = get_psd_in_range((fs, Pxx_stderr), [0, 40])
    
    return f, Pxx_mean, Pxx_stderr

def butter_bandpass_filter(data, lowcut, highcut, order=3):
    nyq = 0.5 * SAMPLING_RATE
    low = lowcut / nyq
    high = highcut / nyq
    sos = signal.butter(order, [low, high], analog=False, btype='bandpass', output='sos')
    y = signal.sosfilt(sos, data)
    return y

def butter_lowpass_filter(data, lowcut, order=3):
    nyq = 0.5 * SAMPLING_RATE
    low = lowcut / nyq
    sos = signal.butter(order, low, analog=False, btype='lowpass', output='sos')
    y = signal.sosfilt(sos, data)
    return y


def butter_highpass_filter(data, highcut, order=3):
    nyq = 0.5 * SAMPLING_RATE
    high = highcut / nyq
    sos = signal.butter(order, high, analog=False, btype='highpass', output='sos')
    y = signal.sosfilt(sos, data)
    return y

def scale_0_to_1 (data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def multitaper_psd (data, NW=3, k=5, resample_freq=1000, show_progress=False):
    # Resample to 1000 Hz to speed up multitaper
    data = signal.resample(data, int(len(data)*resample_freq/SAMPLING_RATE))
    
    # How many seconds for each window (N = fs*duration)
    window = 1
    N = int(resample_freq*window)
    [tapers, eigen] = dpss(N, NW, k)
    
    Pxx_list = []
    
    last_progress = 0
    
    # Proceed through signal advancing in steps of size N
    for idx in range(0, len(data), N//10):
        if show_progress:
            progress = (idx/resample_freq)//60
            if last_progress != progress:
                last_progress = progress
                print('Processed {} mins'.format(progress))
        
        y = data[idx:idx+N]
        
        # For a constant window, pad data with zeros to make sure
        # each window is of length N
        if len(y) < N:
            padding = N - len(y)
            y = np.concatenate( (y, np.zeros(padding)) )
        
        Sk_complex, weights, eigenvalues = pmtm(y, e=eigen, v=tapers, show=False)
        Sk = abs(Sk_complex)**2
        Sk = np.mean(Sk * np.transpose(weights), axis=0)

        Pxx_list.append( Sk[0:N//2] )
        
    # Get list of frequencies to accompany PSD
    dt = 1.0/resample_freq
    f = np.linspace(0.0, 1.0/(2.0*dt), N//2)

    # Average the PSD over all windows
    Pxx = np.mean(Pxx_list, axis=0)
    
    return f, Pxx

def get_psd_in_range (PSD, freqs):
    f, Pxx = PSD
    
    f_low = np.where(f >= freqs[0])[0][0]
    f_high = np.where(f >= freqs[1])[0][0]
    
    return f[f_low:f_high], Pxx[f_low:f_high]

def get_mean_PSD_freq (PSD):
    max_freq = 80    
    f, Pxx = PSD
    
    max_freq_idx = np.where(f <= 80)[0][-1]
    return np.sum(f[:max_freq_idx]*Pxx[:max_freq_idx])/np.sum(Pxx[:max_freq_idx])

def get_relative_power (PSD):
    f, Pxx = PSD
    
    max_idx = np.where(f >= 100)[0][0]
    f = f[:max_idx]
    Pxx = Pxx[:max_idx]
    
    spindle_idxs = np.where(np.logical_and(f > 8, f <= 30))
    spindle_power = np.sum(Pxx[spindle_idxs]) / sum(Pxx)

    gamma_idxs = np.where(np.logical_and(f > 30, f <= 80))
    gamma_power = np.sum(Pxx[gamma_idxs]) / sum(Pxx)

    return spindle_power, gamma_power

def get_maximal_frequency (PSD):
    f, Pxx = PSD
    
    # Discard DC value
    f = f[1:]
    Pxx = Pxx[1:]

    return f[np.argmax(Pxx)]

def imag_coherence(x, y, fs=1.0, window='hann', nperseg=None, noverlap=None, \
              nfft=None, detrend='constant', axis=-1):

    freqs, Pxx = signal.welch(x, fs=fs, window=window, nperseg=nperseg,
                       noverlap=noverlap, nfft=nfft, detrend=detrend,
                       axis=axis)
    _, Pyy = signal.welch(y, fs=fs, window=window, nperseg=nperseg, noverlap=noverlap,
                   nfft=nfft, detrend=detrend, axis=axis)
    _, Pxy = signal.csd(x, y, fs=fs, window=window, nperseg=nperseg,
                 noverlap=noverlap, nfft=nfft, detrend=detrend, axis=axis)

    Cxy = np.abs(np.imag(
        Pxy / (Pxx*Pyy)**0.5
    ))

    return freqs, Cxy

def rms (data):
    square = [d**2 for d in data]
    mean = np.mean(square)
    root = mean**0.5
    return root

######################
# Constant variables #
######################

SAMPLING_RATE    = 30000
STRIATUM_CHANNEL = 2
THALAMUS_CHANNEL = 18
FIG_ROOT         = './figures/'
DATA_RANGE_ALL   = get_slice_from_s(0*60, 60*60) 
DATA_RANGE_FIRST_MIN   = get_slice_from_s(0*60, 1*60)

COLOR_CORTEX   = 'tab:green'
COLOR_THALAMUS = 'tab:purple'
COLOR_STRIATUM = 'tab:orange'