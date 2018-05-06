import matplotlib
matplotlib.use('Agg')  # Or any other X11 back-end   
import numpy as np
import torch.nn as nn
import torch.nn.init as init

  
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy import signal
import sys
import time
import torch
import torch.nn as nn
from torch.autograd import Variable

import h5py
import numpy as np
import os
import pandas as pd
from torch.utils.data import Dataset
import wfdb

# GLOBALS
SIGNALS = ['SaO2',
           'ABD',
           'CHEST',
           'Chin1-Chin2',
           'AIRFLOW',
           'ECG',
           'E1-M2',
           'O2-M1',
           'C4-M1',
           'C3-M2',
           'F3-M2',
           'F4-M1',
           'O1-M2'
          ]

TRAINING_START = 0
TRAINING_SIZE = 30
VALIDATION_START = TRAINING_START + TRAINING_SIZE
VALIDATION_SIZE = 5
HW = 128
SAMPLES_PER_WINDOW = 50
OVERLAP = SAMPLES_PER_WINDOW//2 # offset by half the window
MINUTES = 2
raw_window_size = int(MINUTES*60*200)
WS = raw_window_size + (HW - (raw_window_size + HW) % HW)
LR = 1e-3

print('adjusted window size: {}, num bins: {}'.format(WS, WS//HW))


class SleepDataset(Dataset):
    """Physionet 2018 dataset."""

    def __init__(self, records_file, root_dir, s, f, hanning_window, signals):
        """
        Args:
            records_file (string): Path to the records file.
            root_dir (string): Directory with all the signals.

        """
        self.landmarks_frame = pd.read_csv(records_file)[s:f]
        self.root_dir = root_dir
        self.hw = hanning_window
        self.signals = signals

    @staticmethod
    def to_spectogram(matrix, hw):
        spectograms = []
        for i in range(matrix.shape[1]):
            f, t, Sxx = signal.spectrogram(matrix[:,i], 
                               window=signal.get_window(('exponential', None, 2),hw, False),
                               # window=signal.get_window('hann',hw, False), 
                               fs=200, 
                               scaling='density', 
                               mode='magnitude',
                               noverlap=0
                              )
            spectograms.append(Sxx)
        return np.array(spectograms)

        
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
                
        arous_data = arousals['data']['arousals'].value.ravel()
                
        signal_idxs = np.nonzero(np.in1d(signals.sig_name, self.signals))[0]
        spectrograms = self.to_spectogram(signals.p_signal[:, signal_idxs], self.hw)
#         print(len(arous_data))
        
        chunk_labels = np.split(arous_data[:(len(arous_data)//self.hw)*self.hw].clip(min=0), len(arous_data)//self.hw)
        chunk_labels = np.mean(chunk_labels, axis=1)
        
#         f, a = plt.subplots(14, 1, figsize=(10,10))
#         a[0].plot(chunk_labels)
#         a[0].set_xlim([0, len(chunk_labels)])
#         for i in range(13):
#             a[i+1].imshow(spectrograms[i], aspect='auto')
        
#         plt.show()
        
    
        return spectrograms, chunk_labels

class LSTM_1(nn.Module):

    def __init__(self, window_size, han_size, num_signals):
        super(LSTM_1, self).__init__()
        num_bins = window_size
        self.num_directions = 2
        self.hidden_size = num_bins*2
        self.num_features = 32
                
        self.pad = nn.ZeroPad2d((3, 3, 0, 0))
        
        self.cnns = []
        self.cnns2 = [] 
        for i in range(num_signals):
            self.cnns.append(nn.Conv2d(1, 16, (han_size//2+1, 7)))
        
            init.xavier_uniform(self.cnns[-1].weight, gain=nn.init.calculate_gain('relu'))
            init.constant(self.cnns[-1].bias, 0.1)
            
            self.cnns2.append(nn.Conv2d(16, self.num_features, (1, 7)))
        
            init.xavier_uniform(self.cnns2[-1].weight, gain=nn.init.calculate_gain('relu'))
            init.constant(self.cnns2[-1].bias, 0.1)
        
        self.cnns = nn.ModuleList(self.cnns)
        self.cnns2 = nn.ModuleList(self.cnns2)
#         self.cnn2 = nn.Conv2d(num_signals, self.num_features, (self.num_features, 5))
        
#         init.xavier_uniform(self.cnn2.weight, gain=nn.init.calculate_gain('relu'))
#         init.constant(self.cnn2.bias, 0.1)
        
        self.gru = nn.GRU(self.num_features, self.hidden_size, num_layers=4, bidirectional=True)
        self.lstm = nn.LSTM(self.num_features, self.hidden_size, num_layers=2, bidirectional=True)

        self.ap = nn.AdaptiveMaxPool2d((num_bins, self.num_features))
        
        self.reshape1 = nn.Conv2d(13, 6, 1)
        init.xavier_uniform(self.reshape1.weight, gain=nn.init.calculate_gain('relu'))
        init.constant(self.reshape1.bias, 0.1)
        self.reshape2 = nn.Conv2d(6, 1, 1)
        init.xavier_uniform(self.reshape2.weight, gain=nn.init.calculate_gain('relu'))
        init.constant(self.reshape2.bias, 0.1)
        
        self.fc = nn.Linear(self.num_features*num_bins, num_bins)#self.hidden_size*self.num_directions)
        
        self.out = nn.Linear(num_bins, num_bins)
        
        self.sig = nn.Sigmoid()
        self.relu = nn.ReLU()
#         self.tanh = nn.Tanh()
   
    def init_hidden(self, batch_size):
        # variable of size [num_layers*num_directions, b_sz, hidden_sz]
        return Variable(torch.zeros(4*self.num_directions, batch_size, self.hidden_size)).cuda() 

    def forward(self, x):
#         f, ax = plt.subplots(13, 1, figsize=(10,10))
        
        x = self.pad(x)
        t = []
        tmp = x.cpu().squeeze(0).data.numpy()
        for i in range(x.size()[1]):
#             ax[i].imshow(tmp[i], aspect='auto')
            a = x[:,0,:,:].unsqueeze(0)
            t.append(self.relu(self.cnns2[i](self.pad(self.relu(self.cnns[i](a))))))
#         plt.show()
        x = torch.stack(t).transpose(2, 3).transpose(0,2).squeeze(0)
#         print(x.size())
#         x = self.ap(x)
        x = self.relu(self.reshape1(x))
        x = self.relu(self.reshape2(x))
#         print(x.size())
#         x = x.transpose(1,3).squeeze(0)
#         print(x.size())
#         x = x.transpose(1,2)
#         print(x.size())
#         x = self.pad(x)
#         x = self.relu(self.cnn2(x))
#         x = x.transpose(1,3).squeeze(0)
#         h = self.init_hidden(1)
#         x, h = self.gru(x)
#         print(x.size())
#         print(x.view(-1))
        x = self.fc(x.view(-1))
        x = self.out(x)
        return self.sig(x)




root = os.getcwd()
image_dir = os.path.join(root, 'outputs', '{}WS_{}HS_{:.0E}SGD-BCE'.format(WS, HW, LR))
if not os.path.isdir(image_dir):
    os.makedirs(image_dir)


#TODO add torch.save(the_model.state_dict(), PATH) this to save the best models weights

train_dataset = SleepDataset('/beegfs/ga4493/projects/groupb/data/training/RECORDS', 
                             '/beegfs/ga4493/projects/groupb/data/training/', TRAINING_START, TRAINING_START+TRAINING_SIZE, HW, SIGNALS)

train_loaders = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=1,
                                           shuffle=True)

test_dataset = SleepDataset('/beegfs/ga4493/projects/groupb/data/training/RECORDS', 
                                '/beegfs/ga4493/projects/groupb/data/training/', VALIDATION_START, VALIDATION_START + VALIDATION_SIZE, HW, SIGNALS)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                          batch_size=1, 
                                          shuffle=False)

# d
model = LSTM_1(SAMPLES_PER_WINDOW, HW, len(SIGNALS))

if torch.cuda.is_available():
    print('using cuda')
    model.cuda()

criterion = nn.BCELoss()

optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=1e-3)

losses = []
v_losses = []
accuracy = []
v_accuracy = []
l = None
best_validation = None
fh = open(os.path.join(image_dir, 'Losses{:.0E}_lr-{}_ws-{}_hanw.txt'.format(LR, WS, HW)), 'w')
for epoch in range(200):
    loss_t = 0.0
    acc_t = 0.0
    count_t = 0
    start_time = time.time()
    val_l = None
    v_out = None
    v_all = []
    for c, (spectrograms, labels) in enumerate(train_loaders):
        np.random.seed(12345 + c)
        for i in range(labels.size()[1]//OVERLAP-1):
            start_pos = (i*OVERLAP)
            end_pos = (i+1)*OVERLAP+OVERLAP
            optimizer.zero_grad()
            l = labels[:,start_pos:end_pos].type(torch.FloatTensor)
            rand_val = np.random.rand(1)[0]
            if l.mean() > 0.05 or rand_val < 0.1:
                inp_subs = spectrograms[:,:,:,start_pos:end_pos].type(torch.FloatTensor)


                if torch.cuda.is_available():
                    l = l.cuda()
                    inp_subs = inp_subs.cuda()
                l = Variable(l)
                inp_subs = Variable(inp_subs)

                output = model(inp_subs)

                loss = criterion(output, l.view(-1))
    #             print(loss)

                loss_t += loss.data[0]

                comparison = (output.cpu().data.numpy().ravel() > 0.5) == (l.cpu().data.numpy())
                acc_t += comparison.sum() / SAMPLES_PER_WINDOW

                count_t += 1

                loss.backward()
                optimizer.step()
    losses.append(loss_t/count_t)
    accuracy.append(acc_t/count_t)
#     #####
    loss_v = 0.0
    acc_v = 0.0
    count_v = 0
    for c_tst, (spectrograms_tst, labels_tst) in enumerate(test_loader):
        np.random.seed(12345 + c_tst)
        for i in range(labels_tst.size()[1]//OVERLAP-1):
            start_pos_tst = (i*OVERLAP)
            end_pos_tst = (i+1)*OVERLAP+OVERLAP
            l_tst = labels_tst[:,start_pos_tst:end_pos_tst].type(torch.FloatTensor)
            rand_val = np.random.rand(1)[0]
            if l_tst.mean() > 0.05 or rand_val < 0.1:
                rand_val = np.random.rand(1)[0]
                inp_subs_tst = spectrograms_tst[:,:,:,start_pos_tst:end_pos_tst].type(torch.FloatTensor)
                if torch.cuda.is_available():
                    l_tst = l_tst.cuda()
                    inp_subs_tst = inp_subs_tst.cuda()
                    
                l_tst = Variable(l_tst)
                inp_subs_tst = Variable(inp_subs_tst)
                
                output_tst = model(inp_subs_tst)
                loss_tst = criterion(output_tst, l_tst.view(-1))

                loss_v += loss_tst.data[0]
                count_v += 1
                
                comparison = (output_tst.cpu().data.numpy().ravel() > 0.5) == (l_tst.cpu().data.numpy())
                acc_v += comparison.sum() / SAMPLES_PER_WINDOW
            
    v_losses.append(loss_v/count_v)
    v_accuracy.append(acc_v/count_v)
    if best_validation == None or v_losses[-1] < best_validation:
        print('{} better than {}, saving model'.format(v_losses[-1], best_validation))
        fh.write('{} better than {}, saving model\n'.format(v_losses[-1], best_validation))
        torch.save(model.state_dict(), os.path.join(image_dir, 'best_params.pt'))
        best_validation = v_losses[-1]
    fh.write('#'*45+'\n')
    fh.write('# epoch  - {:>10} | time(s) -{:>10.2f} #\n'.format(epoch, time.time() - start_time))
    fh.write('# T loss - {:>10.2f} | V loss - {:>10.2f} #\n'.format(loss_t/count_t, loss_v/count_v))
    fh.flush()
    print('#'*45)
    print('# epoch  - {:>10} | time(s) -{:>10.2f} #'.format(epoch, time.time() - start_time))
    print('# T loss - {:>10.6f} | V loss - {:>10.6f} #'.format(loss_t/count_t, loss_v/count_v))
    print('# T acc  - {:>10.6f} | V acc  - {:>10.6f} #'.format(acc_t/count_t, acc_v/count_v))
print('#'*45)
fh.write('#'*45)
fh.close()




