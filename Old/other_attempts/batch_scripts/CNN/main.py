from DataLoader import SleepDataset, SleepDatasetTest
import matplotlib
matplotlib.use('Agg')  # Or any other X11 back-end   
import matplotlib.pyplot as plt
from Model_v1 import Model_V1
import numpy as np
import os
from scipy import signal
import time
import torch
import torch.nn as nn
from torch.autograd import Variable

window_size = 36000*2
hanning_window = 256
learning_rate = 1e-3

#TODO add torch.save(the_model.state_dict(), PATH) this to save the best models weights

root = os.getcwd()

image_dir = os.path.join(root, 'outputs', 'all_positive-{}WS_{}HS_{:.0E}LR-layers-16-16-16-Adam_sample_avaerage-F'.format(window_size, hanning_window, learning_rate))

if not os.path.isdir(image_dir):
    os.makedirs(image_dir)

def to_spectogram(matrix):
    spectograms = []
    for i in range(matrix.size()[2]):
        f, t, Sxx = signal.spectrogram(matrix[0,:,i].numpy(),
                           window=signal.get_window('hann',hanning_window, False),
                           fs=200,
                           scaling='density',
                           mode='magnitude',
                           noverlap=0
                          )
        spectograms.append(Sxx)
    if torch.cuda.is_available():
        return torch.FloatTensor(spectograms).unsqueeze(0).cuda()
    else:
        return torch.FloatTensor(spectograms).unsqueeze(0)


train_dataset = SleepDataset('/beegfs/ga4493/projects/groupb/data/training/RECORDS', '/beegfs/ga4493/projects/groupb/data/training/', 100, 150)

train_loaders = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=1,
                                           shuffle=True)

model_v1 = Model_V1(window_size, hanning_window, window_size)

if torch.cuda.is_available():
    print('using cuda')
    model_v1.cuda()
# TODO change this to BCEWithLogitsLoss
# TODO CHANGE size_average to false, taking the mean is probably not a good idea :(
criterion = nn.BCEWithLogitsLoss(size_average=False)

optimizer = torch.optim.Adam(model_v1.parameters(), lr=learning_rate)#, momentum=0.9)#, weight_decay=1e-3)  
sig = nn.Sigmoid()
test_dataset = SleepDatasetTest('/beegfs/ga4493/projects/groupb/data/training/RECORDS', '/beegfs/ga4493/projects/groupb/data/training/', 0, 10)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                          batch_size=1, 
                                          shuffle=False)

# i, ((data, cent), v_l) = next(enumerate(test_loader))
losses = []
v_losses = []
l = None
for epoch in range(50):
    loss_t = 0.0
    start_time = time.time()
    for i, ((all_data, centers), labels) in enumerate(train_loaders):
        for i in range((all_data.size()[1]//window_size) - 1):
            inp_subs = Variable(to_spectogram(all_data[:,i*window_size:(i+1)*window_size,]))
            l = None
            if torch.cuda.is_available():
                l = Variable(labels[0, i*window_size:(i+1)*window_size].type(torch.FloatTensor).cuda())
            else:
                l = Variable(labels[0, i*window_size:(i+1)*window_size].type(torch.FloatTensor))
            output = model_v1(inp_subs)
            loss = criterion(output.view(-1), l)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_t += loss.data[0] / len(centers)

    losses.append(loss_t)
    print('epoch', epoch, 'loss', loss_t, 'time', time.time() - start_time)
    for j, ((data, cent), v_l) in enumerate(test_loader):
        val_l = None
        v_out = None
        v_all = []
        for i in range((data.size()[1]//window_size)//2 - 1):
            inp_subs = Variable(to_spectogram(data[:,i*window_size:(i+1)*window_size,]))
            v_out = model_v1(inp_subs)
            v_out = sig(v_out)
            v_all = np.append(v_all, v_out.cpu().data[0].numpy())
        fig = plt.figure(figsize=(20, 10))
        plt.plot(v_all)
        plt.plot(v_l.numpy()[0][:len(v_l.numpy()[0])//2] > 0, alpha=0.3)
        plt.plot(v_l.numpy()[0][:len(v_l.numpy()[0])//2] < 0, alpha=0.3)
        plt.title('({}_lr-{}_ws-{}_hanw) loss'.format(learning_rate, window_size, hanning_window))
        fig.savefig(os.path.join(image_dir, 'epoch-{}_Test{:.0E}_lr-{}_ws-{}_hanw_{}.png'.format(epoch, learning_rate, window_size, hanning_window, j)))
        plt.close(fig)

fig = plt.figure(figsize=(10, 10))
plt.plot(losses)
fig.savefig(os.path.join(image_dir, 'Losses{:.0E}_lr-{}_ws-{}_hanw.png'.format(learning_rate, window_size, hanning_window)))
plt.close(fig)
