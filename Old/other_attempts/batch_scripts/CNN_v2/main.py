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
        interested = []
        for i in range(13):
            if signals.sig_name[i] in ['SaO2', 'ABD', 'F4-M1', 'C4-M1', 'O2-M1', 'AIRFLOW']:
                interested.append(i)
#         POI = arousal_centers
        sample =  ((signals.p_signal[:,interested], POI), arousals['data']['arousals'].value.ravel())
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
        interested = []
        for i in range(13):
            if signals.sig_name[i] in ['SaO2', 'ABD', 'F4-M1', 'C4-M1', 'O2-M1', 'AIRFLOW']:
                interested.append(i)
#         POI = arousal_centers
        tmp = arousals['data']['arousals'].value.ravel()
        tmp[tmp < 0] = 0.5
        sample =  ((signals.p_signal[:,interested], POI), tmp)
        return sample


class Model_V3(nn.Module):

    def __init__(self, window_size, han_size, slice_size):
        super(Model_V3, self).__init__()
        self.cnn1 = nn.Conv2d(6, 16, 3, padding=1)
        # self.cnn1 = nn.Conv2d(6, 16, 3, padding=1)
        init.xavier_uniform(self.cnn1.weight, gain=nn.init.calculate_gain('relu'))
        init.constant(self.cnn1.bias, 0.1)
        self.cnn2 = nn.Conv2d(16, 16, 3, padding=1)
        # self.cnn2 = nn.Conv2d(16, 16, 3, padding=1)
        init.xavier_uniform(self.cnn2.weight, gain=nn.init.calculate_gain('relu'))
        init.constant(self.cnn2.bias, 0.1)
        self.cnn3 = nn.Conv2d(16, 16, 3, padding=1)
        # self.cnn3 = nn.Conv2d(16, 16, 3, padding=1)
        init.xavier_uniform(self.cnn3.weight, gain=nn.init.calculate_gain('relu'))
        init.constant(self.cnn3.bias, 0.1)
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        
#         self.dropout2 = nn.Dropout2d()
        
#         self.dropout = nn.Dropout()
        
#         self.hidden = nn.Linear(16*17*64, 10000)
        out_dim = ((han_size//2+1)//8)*((window_size//han_size)//8)*16
        self.output = nn.Linear(out_dim, slice_size)
        
        self.sigmoid = nn.Sigmoid()
        
   
    def forward(self, x):
#         x = self.relu(self.pool(self.dropout2(self.cnn1(x))))
#         x = self.relu(self.pool(self.dropout2(self.cnn2(x))))
#         x = self.relu(self.pool(self.dropout2(self.cnn3(x))))
#         print(x.size())
        x = self.relu(self.pool(self.cnn1(x)))
        x = self.relu(self.pool(self.cnn2(x)))
        x = self.relu(self.pool(self.cnn3(x)))
#         print(x.size())
        x = x.view(1, -1)
#         print(x.size())
#         x = self.dropout(x)
#         x = self.relu(self.hidden(x))
        x = self.output(x)
        return self.sigmoid(x)




window_size = int(sys.argv[1]) # 36000 worked pretty well
hanning_window = int(sys.argv[2])#1024 # 256 also w
learning_rate = float(sys.argv[3])

root = os.getcwd()
image_dir = os.path.join(root, 'outputs', 'negative_to_0-5--10samples-{}WS_{}HS_{:.0E}LR-layers-16-16-16-Adam-wd1e-3-MSE'.format(window_size, hanning_window, learning_rate))
if not os.path.isdir(image_dir):
    os.makedirs(image_dir)

def to_spectogram(matrix):
    spectograms = []
    for i in range(all_data.size()[2]):
        f, t, Sxx = signal.spectrogram(matrix[0,:,i].numpy(), 
                           window=signal.get_window('hann',hanning_window, False), 
                           fs=200, 
                           scaling='density', 
                           mode='magnitude',
                           noverlap=0
                          )
        if (Sxx.min() != 0 or Sxx.max() != 0):
            spectograms.append((Sxx - Sxx.min()) / (Sxx.max() - Sxx.min()))
        else:
            spectograms.append(Sxx)
    return torch.FloatTensor(spectograms).unsqueeze(0).cuda()

#TODO add torch.save(the_model.state_dict(), PATH) this to save the best models weights

train_dataset = SleepDataset('/beegfs/ga4493/projects/groupb/data/training/RECORDS', 
                             '/beegfs/ga4493/projects/groupb/data/training/', 20, 30)

train_loaders = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=1,
                                           shuffle=True)

model = Model_V3(window_size, hanning_window, window_size)

if torch.cuda.is_available():
    print('using cuda')
    model.cuda()
# TODO change this to BCEWithLogitsLoss
# criterion = nn.BCEWithLogitsLoss(size_average=False)
criterion = nn.MSELoss(size_average=False)
# criterion = nn.BCELoss(size_average=False)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-3)
# optimizer = torch.optim.Adadelta(model.parameters())
# optimizer = torch.optim.RMSprop(model.parameters(), momentum=0.9)
# optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, nesterov=True)
sig = nn.Sigmoid()
test_dataset = SleepDatasetTest('/beegfs/ga4493/projects/groupb/data/training/RECORDS', 
                                '/beegfs/ga4493/projects/groupb/data/training/', 0, 10)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                          batch_size=1, 
                                          shuffle=False)

# i, ((data, cent), v_l) = next(enumerate(test_loader))
losses = []
v_losses = []
l = None
best_validation = None
fh = open(os.path.join(image_dir, 'Losses{:.0E}_lr-{}_ws-{}_hanw.txt'.format(learning_rate, window_size, hanning_window)), 'w')
for epoch in range(50):
    loss_t = 0.0
    count_t = 0
    start_time = time.time()
    val_l = None
    v_out = None
    v_all = []
    for c, ((all_data, centers), labels) in enumerate(train_loaders):
        for i in range((all_data.size()[1]//window_size) - 1):
            inp_subs = Variable(to_spectogram(all_data[:,i*window_size:(i+1)*window_size,]))
            l = None
            if torch.cuda.is_available():
                l = Variable(labels[0, i*window_size:(i+1)*window_size].type(torch.FloatTensor).clamp(min=0).cuda())
                # l = Variable(labels[0, i*window_size:(i+1)*window_size].type(torch.FloatTensor).abs().cuda())
            else:
                l = Variable(labels[0, i*window_size:(i+1)*window_size].type(torch.FloatTensor).clamp(min=0))
                # l = Variable(labels[0, i*window_size:(i+1)*window_size].type(torch.FloatTensor).abs())
            output = model(inp_subs)
            loss = criterion(output.view(-1), l)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_t += loss.data[0]
            count_t += 1
    losses.append(loss_t/count_t)
    
    loss_v = 0.0
    count_v = 0
    for c, ((data, cent), v_l) in enumerate(test_loader):
        for i in range((data.size()[1]//window_size) - 1):
            inp_subs = Variable(to_spectogram(data[:,i*window_size:(i+1)*window_size,]))
            l = None
            if torch.cuda.is_available():
                l = Variable(v_l[0, i*window_size:(i+1)*window_size].type(torch.FloatTensor).clamp(min=0).cuda())
                # l = Variable(v_l[0, i*window_size:(i+1)*window_size].type(torch.FloatTensor).abs().cuda())
            else:
                l = Variable(v_l[0, i*window_size:(i+1)*window_size].type(torch.FloatTensor).clamp(min=0))
                # l = Variable(v_l[0, i*window_size:(i+1)*window_size].type(torch.FloatTensor).abs())
            output = model(inp_subs)
            loss = criterion(output.view(-1), l)

            loss_v += loss.data[0]
            count_v += 1
    v_losses.append(loss_v/count_v)
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
    print('# T loss - {:>10.2f} | V loss - {:>10.2f} #'.format(loss_t/count_t, loss_v/count_v))
print('#'*45)
fh.write('#'*45)
fh.close()
fig = plt.figure(figsize=(10, 10))
plt.plot(losses)
plt.plot(v_losses)
fig.savefig(os.path.join(image_dir, 'Losses{:.0E}_lr-{}_ws-{}_hanw.png'.format(learning_rate, window_size, hanning_window)))
plt.close(fig)
# plt.show()
model.load_state_dict(torch.load(os.path.join(image_dir, 'best_params.pt'), map_location=lambda storage, loc: storage))
# for j, ((data, cent), v_l) in enumerate(train_loaders):
    # val_l = None
    # v_out = None
    # v_all = []
    # for i in range((data.size()[1]//window_size) - 1):
        # inp_subs = Variable(to_spectogram(data[:,i*window_size:(i+1)*window_size,]))
        # v_out = model(inp_subs)
# #         v_out = sig(v_out)
        # v_all = np.append(v_all, v_out.cpu().data[0].numpy())
    # fig = plt.figure(figsize=(20, 10))
    # plt.plot(v_all)
    # plt.plot((v_l.numpy()[0][:len(v_l.numpy()[0])] > 0).astype(float)*1.1, alpha=0.3)
    # plt.plot((v_l.numpy()[0][:len(v_l.numpy()[0])] < 0).astype(float)*1.1, alpha=0.3)
    # plt.title('Train({}_lr-{}_ws-{}_hanw) loss'.format(learning_rate, window_size, hanning_window))
    # # plt.show()
    # fig.savefig(os.path.join(image_dir, 'epoch-{}_Train{:.0E}_lr-{}_ws-{}_hanw_{}.png'.format(epoch, learning_rate, window_size, hanning_window, j)))
    # plt.close(fig)

for j, ((data, cent), v_l) in enumerate(test_loader):
    val_l = None
    v_out = None
    v_all = []
    for i in range((data.size()[1]//window_size) - 1):
        inp_subs = Variable(to_spectogram(data[:,i*window_size:(i+1)*window_size,]))
        v_out = model(inp_subs)
#         v_out = sig(v_out)
        v_all = np.append(v_all, v_out.cpu().data[0].numpy())
    fig = plt.figure(figsize=(20, 10))
    plt.plot(v_all)
    plt.plot((v_l.numpy()[0][:len(v_l.numpy()[0])] > 0).astype(float)*1.1, alpha=0.3)
    plt.plot((v_l.numpy()[0][:len(v_l.numpy()[0])] < 0).astype(float)*1.1, alpha=0.3)
    plt.title('Test({}_lr-{}_ws-{}_hanw) loss'.format(learning_rate, window_size, hanning_window))
    # plt.show()
    fig.savefig(os.path.join(image_dir, 'epoch-{}_Test{:.0E}_lr-{}_ws-{}_hanw_{}.png'.format(epoch, learning_rate, window_size, hanning_window, j)))
    plt.close(fig)
