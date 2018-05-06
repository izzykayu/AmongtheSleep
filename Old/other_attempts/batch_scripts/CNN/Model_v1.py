import matplotlib
matplotlib.use('Agg')  # Or any other X11 back-end   
import numpy as np
import torch.nn as nn
import torch.nn.init as init


class Model_V1(nn.Module):
    def __init__(self, window_size, han_size, slice_size):
        super(Model_V1, self).__init__()
        self.cnn1 = nn.Conv2d(6, 16, 7, padding=3)
        init.xavier_uniform(self.cnn1.weight, gain=np.sqrt(2.0))
        init.constant(self.cnn1.bias, 0.1)
        self.cnn2 = nn.Conv2d(16, 16, 5, padding=2)
        init.xavier_uniform(self.cnn2.weight, gain=np.sqrt(2.0))
        init.constant(self.cnn2.bias, 0.1)
        self.cnn3 = nn.Conv2d(16, 16, 3, padding=1)
        init.xavier_uniform(self.cnn3.weight, gain=np.sqrt(2.0))
        init.constant(self.cnn3.bias, 0.1)
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
#         self.dropout2 = nn.Dropout2d()
#         self.dropout = nn.Dropout()
#         self.hidden = nn.Linear(16*17*64, 10000)
        out_dim = ((han_size//2+1)//8)*((window_size//han_size)//8)*16
        self.output = nn.Linear(out_dim, slice_size)
        # print((han_size//2+1)//8, (window_size//han_size)//8, 16, out_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
#         x = self.relu(self.pool(self.dropout2(self.cnn1(x))))
#         x = self.relu(self.pool(self.dropout2(self.cnn2(x))))
#         x = self.relu(self.pool(self.dropout2(self.cnn3(x))))
        x = self.relu(self.pool(self.cnn1(x)))
        x = self.relu(self.pool(self.cnn2(x)))
        x = self.relu(self.pool(self.cnn3(x)))
        # print(x.size())
        x = x.view(1, -1)
        # print(x.size())
#         x = self.dropout(x)
#         x = self.relu(self.hidden(x))
        x = self.output(x)
        return x
