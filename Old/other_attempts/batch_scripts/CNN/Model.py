import matplotlib
matplotlib.use('Agg')  # Or any other X11 back-end   
import numpy as np
import torch.nn as nn
import torch.nn.init as init

class Model_V1(nn.Module):

    def __init__(self, slice_size):
        super(Model_V1, self).__init__()
        self.cnn1 = nn.Conv2d(6, 16, 3, padding=1)
        init.xavier_uniform(self.cnn1.weight, gain=np.sqrt(2.0))
        init.constant(self.cnn1.bias, 0.1)
        self.cnn2 = nn.Conv2d(16, 16, 3, padding=1)
        init.xavier_uniform(self.cnn2.weight, gain=np.sqrt(2.0))
        init.constant(self.cnn2.bias, 0.1)
        self.cnn3 = nn.Conv2d(16, 16, 3, padding=1)
        init.xavier_uniform(self.cnn3.weight, gain=np.sqrt(2.0))
        init.constant(self.cnn3.bias, 0.1)
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.LeakyReLU()
        
        self.dropout2 = nn.Dropout2d(p=0.01)
        
#         self.dropout = nn.Dropout()
        
#         self.hidden = nn.Linear(16*17*64, 10000)
        
        self.output = nn.Linear(16*17*16, slice_size)
        
        self.sigmoid = nn.Sigmoid()
        
   
    def forward(self, x):
        # x = self.relu(self.pool(self.dropout2(self.cnn1(x))))
        x = self.relu(self.pool(self.cnn1(x)))
        x = self.relu(self.pool(self.dropout2(self.cnn2(x))))
        # x = self.relu(self.pool(self.cnn2(x)))
        x = self.relu(self.pool(self.dropout2(self.cnn3(x))))
        # x = self.relu(self.pool(self.cnn3(x)))
#         print(x.size())

#         print(x.size())
        x = x.view(1, -1)
#         print(x.size())
#         x = self.dropout(x)
#         x = self.relu(self.hidden(x))
        x = self.output(x)
        return x
