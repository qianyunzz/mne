import os
import numpy as np
import mne

from mne import read_source_estimate
from mne.datasets import sample



print(__doc__)

# sample_dir_raw = sample.data_path()
# sample_dir_raw.replace('\\', '7')
# sample_dir_raw = 'D:/code/data/MNE/MNE-sample-data'
# sample_dir = os.path.join(sample_dir_raw, 'MEG', 'sample')
# subjects_dir = os.path.join(sample_dir_raw, 'subjects')
sample_dir = 'data/MNE/MNE-sample-data/MEG/sample'
subjects_dir = 'D:/code/data/MNE/MNE-sample-data/subjects'
# fname_stc = os.path.join(sample_dir, 'sample_audvis-meg')
fname_stc = 'D:/code/data/MNE/MNE-sample-data/MEG/sample/sample_audvis-meg'
stc = read_source_estimate(fname_stc, subject='sample')

# Define plotting parameters
surfer_kwargs = dict(hemi='lh', subjects_dir=subjects_dir,
                     clim=dict(kind='value', lims=[8, 12, 15]), views='lateral',
                     initial_time=0.09, time_unit='s', size=(800, 800),
                     smoothing_steps=5)

# plot surface
# youwenti
brain = stc.plot(**surfer_kwargs)

# add title
brain.add_text(0.1, 0.9, 'SourceEstimate', 'title', font_size=16)
# brain.maximize_window()
# brain.sleep(5)


shape = stc.data.shape

print(shape)
# print('The data has %s vertex locations with %s sample points each.' % shape)