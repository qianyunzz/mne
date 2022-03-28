import os.path as op
import numpy as np
import matplotlib.pyplot as plt

import mne
from mne.datasets import sample
from mne.minimum_norm import make_inverse_operator, apply_inverse


data_path = 'data/MNE/MNE-sample-data/'
raw_fname = op.join(data_path, 'MEG', 'sample', 'sample_audvis_filt-0-40_raw.fif')

raw = mne.io.read_raw_fif(raw_fname)
events = mne.find_events(raw, stim_channel='STI 014')

event_id = dict(aud_1=1)
tmin = -0.2
tmax = 0.5
raw.info['bads'] = ['MEG 2443', 'EEG 053']
baseline = (None, 0)
reject = dict(eeg=40e-6)  # grad=4000e-13, mag=4e-12, eeg=40e-6, eog=250e-6

epochs = mne.Epochs(raw, events, event_id, tmin, tmax, proj=True, picks=('eeg'), baseline=baseline, reject=reject)

# 计算噪声协方差矩阵
noise_cov = mne.compute_covariance(epochs, tmax=0., method=['shrunk', 'empirical'], rank=None, verbose=True) # method='empirical', 'diagonal_fixed', 'shrunk', 'oas', 'ledoit_wolf', 'factor_analysis', 'shrinkage', and 'pca'

fig_cov, fig_spectra = mne.viz.plot_cov(noise_cov, raw.info)

# 计算诱发反应
evoked = epochs.average().pick('eeg')
evoked.plot(time_unit='s')
evoked.plot_topomap(times=np.linspace(0.05, 0.15, 5), ch_type='eeg', time_unit='s')  # ch_type= ‘mag’ | ‘grad’ | ‘planar1’ | ‘planar2’ | ‘eeg’ | None

evoked.plot_white(noise_cov, time_unit='s')
del epochs, raw  # to save memory

# 逆问题
fname_fwd = data_path + '/MEG/sample/sample_audvis-eeg-oct-6-fwd.fif'
fwd = mne.read_forward_solution(fname_fwd)

# MEG逆算子
inverse_operator = make_inverse_operator(evoked.info, fwd, noise_cov, loose=0.2, depth=2) # depth|meg:0.2,eeg:2-5
del fwd

# 计算逆解
method = "dSPM"    # "dSPM"\"sLORETA"\"eLORETA"\"MNE"
snr = 3.
lambda2 = 1. / snr ** 2
stc, residual = apply_inverse(evoked, inverse_operator, lambda2,
                              method=method, pick_ori=None,
                              return_residual=True, verbose=True)

fig1, ax = plt.subplots()
ax.plot(1e3 * stc.times, stc.data[::100, :].T)
ax.set(xlabel='time (ms)', ylabel='%s value' % method)

fig2, axes = plt.subplots(2, 1)
evoked.plot(axes=axes)
for ax in axes:
    for text in list(ax.texts):
        text.remove()
    for line in ax.lines:
        line.set_color('skyblue')
residual.plot(axes=axes)
fig2.savefig('./eeg_residual.png')


vertno_max, time_max = stc.get_peak(hemi='rh')

subjects_dir = data_path + '/subjects'
surfer_kwargs = dict(
    hemi='rh', subjects_dir=subjects_dir,
    clim=dict(kind='value', lims=[8, 12, 15]), views='lateral',
    initial_time=time_max, time_unit='s', size=(400, 400), smoothing_steps=10)
brain = stc.plot(**surfer_kwargs)

brain.add_foci(vertno_max, coords_as_verts=True, hemi='rh', color='blue',
               scale_factor=0.6, alpha=0.5)
brain.add_text(0.1, 0.9, str(method) + '(plus location of maximal activation)', 'title',
               font_size=14)
