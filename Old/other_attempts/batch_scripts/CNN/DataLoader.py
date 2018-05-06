import matplotlib
matplotlib.use('Agg')  # Or any other X11 back-end   
import h5py
import numpy as np
import os
import pandas as pd
from torch.utils.data import Dataset
import wfdb


class SleepDataset(Dataset):
    """Physionet 2018 dataset."""

    def __init__(self, records_file, root_dir, s, f):
        """
        Args:
            records_file (string): Path to the records file.
            root_dir (string): Directory with all the signals.

        """
        self.landmarks_frame = pd.read_csv(records_file)[s:f]
        self.root_dir = root_dir

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        folder_name = os.path.join(self.root_dir,
                                self.landmarks_frame.iloc[idx, 0])
        file_name = self.landmarks_frame.iloc[idx, 0]
#         print(file_name)
#         print(folder_name)
#         file_name='tr03-0005/'
#         folder_name='../data/training/tr03-0005/'
        signals = wfdb.rdrecord(os.path.join(folder_name, file_name[:-1]))
        arousals = h5py.File(os.path.join(folder_name, file_name[:-1] + '-arousal.mat'), 'r')
        tst_ann = wfdb.rdann(os.path.join(folder_name, file_name[:-1]), 'arousal')
        
        POI = []
        for typ in ['arousal_rera', 'resp_hypopnea', 'resp_centralapnea']:
            start_idx = np.where(np.array(tst_ann.aux_note) == '('+typ)
            end_idx = np.where(np.array(tst_ann.aux_note) == typ+')')

            _starts = tst_ann.sample[start_idx]
            _ends = tst_ann.sample[end_idx]

            _width = np.subtract(_ends, _starts)
            _centers = _starts + _width//2
            POI = np.append(POI, _centers)
        
        W = tst_ann.sample[np.where(np.array(tst_ann.aux_note) == 'W')]
        N1 = tst_ann.sample[np.where(np.array(tst_ann.aux_note) == 'N1')]
        N2 = tst_ann.sample[np.where(np.array(tst_ann.aux_note) == 'N2')]
        N3 = tst_ann.sample[np.where(np.array(tst_ann.aux_note) == 'N3')]
        
       
        POI = np.append(POI, W)
        POI = np.append(POI, N1)
        POI = np.append(POI, N2)
        POI = np.append(POI, N3)
        np.random.shuffle(POI)
#         POI = arousal_centers
        interested = []
        for i in range(13):
            if signals.sig_name[i] in ['SaO2', 'ABD', 'F4-M1', 'C4-M1', 'O2-M1', 'AIRFLOW']:
                interested.append(i)
        # sample =  ((signals.p_signal[:,interested], POI), arousals['data']['arousals'].value.clip(min=0).ravel())
        # NOTE This makes all the non judged regions positive, because it will up the number of positive cases, and the regions arent scored so who cares
        sample =  ((signals.p_signal[:,interested], POI), np.absolute(arousals['data']['arousals'].value.ravel()))
        return sample

class SleepDatasetTest(Dataset):
    """Physionet 2018 dataset."""

    def __init__(self, records_file, root_dir, s, f):
        """
        Args:
            records_file (string): Path to the records file.
            root_dir (string): Directory with all the signals.

        """
        self.landmarks_frame = pd.read_csv(records_file)[s:f]
        self.root_dir = root_dir

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        folder_name = os.path.join(self.root_dir,
                                self.landmarks_frame.iloc[idx, 0])
        file_name = self.landmarks_frame.iloc[idx, 0]
#         print(file_name)
#         print(folder_name)
#         file_name='tr03-0005/'
#         folder_name='../data/training/tr03-0005/'
        signals = wfdb.rdrecord(os.path.join(folder_name, file_name[:-1]))
        arousals = h5py.File(os.path.join(folder_name, file_name[:-1] + '-arousal.mat'), 'r')
        tst_ann = wfdb.rdann(os.path.join(folder_name, file_name[:-1]), 'arousal')
        
        POI = []
        for typ in ['arousal_rera', 'resp_hypopnea', 'resp_centralapnea']:
            start_idx = np.where(np.array(tst_ann.aux_note) == '('+typ)
            end_idx = np.where(np.array(tst_ann.aux_note) == typ+')')

            _starts = tst_ann.sample[start_idx]
            _ends = tst_ann.sample[end_idx]

            _width = np.subtract(_ends, _starts)
            _centers = _starts + _width//2
            POI = np.append(POI, _centers)
        
        W = tst_ann.sample[np.where(np.array(tst_ann.aux_note) == 'W')]
        N1 = tst_ann.sample[np.where(np.array(tst_ann.aux_note) == 'N1')]
        N2 = tst_ann.sample[np.where(np.array(tst_ann.aux_note) == 'N2')]
        N3 = tst_ann.sample[np.where(np.array(tst_ann.aux_note) == 'N3')]
        
       
        POI = np.append(POI, W)
        POI = np.append(POI, N1)
        POI = np.append(POI, N2)
        POI = np.append(POI, N3)
        np.random.shuffle(POI)
#         POI = arousal_centers
        interested = []
        for i in range(13):
            if signals.sig_name[i] in ['SaO2', 'ABD', 'F4-M1', 'C4-M1', 'O2-M1', 'AIRFLOW']:
                interested.append(i)
        sample =  ((signals.p_signal[:,interested], POI), arousals['data']['arousals'].value.ravel())
        return sample

if __name__ in "__main__":
    a = SleepDataset('/beegfs/ga4493/projects/groupb/data/training/RECORDS', '/beegfs/ga4493/projects/groupb//data/training/', 0, 1)
    print(len(a))
    for i in range(1):
        samp = a[i]
