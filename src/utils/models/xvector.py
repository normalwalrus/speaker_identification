import torch
import torch.nn as nn
from torch.nn import functional as F


class simpleTDNN(nn.Module):

    def __init__(self, numSpkrs, p_dropout):
        super(simpleTDNN, self).__init__()
        self.tdnn1 = nn.Conv1d(in_channels=30, out_channels=128, kernel_size=5, dilation=1)
        self.bn_tdnn1 = nn.BatchNorm1d(128, momentum=0.1, affine=False)
        self.dropout_tdnn1 = nn.Dropout(p=p_dropout)

        self.tdnn2 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=5, dilation=2)
        self.bn_tdnn2 = nn.BatchNorm1d(128, momentum=0.1, affine=False)
        self.dropout_tdnn2 = nn.Dropout(p=p_dropout)

        self.tdnn3 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=1, dilation=1)
        self.bn_tdnn3 = nn.BatchNorm1d(128, momentum=0.1, affine=False)
        self.dropout_tdnn3 = nn.Dropout(p=p_dropout)

        self.fc1 = nn.Linear(2*128,128)
        self.bn_fc1 = nn.BatchNorm1d(128, momentum=0.1, affine=False)
        self.dropout_fc1 = nn.Dropout(p=p_dropout)

        self.fc2 = nn.Linear(128,64)
        self.bn_fc2 = nn.BatchNorm1d(64, momentum=0.1, affine=False)
        self.dropout_fc2 = nn.Dropout(p=p_dropout)

        self.fc3 = nn.Linear(64,numSpkrs)

    def forward(self, x, eps):
        # Note: x must be (batch_size, feat_dim, chunk_len)

        x = self.dropout_tdnn1(self.bn_tdnn1(F.relu(self.tdnn1(x))))
        x = self.dropout_tdnn2(self.bn_tdnn2(F.relu(self.tdnn2(x))))
        x = self.dropout_tdnn3(self.bn_tdnn3(F.relu(self.tdnn3(x))))

        if self.training:
            x = x + torch.randn(x.size()).cuda()*eps
        stats = torch.cat((x.mean(dim=2), x.std(dim=2)), dim=1)
        x = self.dropout_fc1(self.bn_fc1(F.relu(self.fc1(stats))))
        x = self.dropout_fc2(self.bn_fc2(F.relu(self.fc2(x))))
        x = self.fc3(x)
        return x


class xvecTDNN(nn.Module):

    def __init__(self, numSpkrs, p_dropout, inchannel):
        super(xvecTDNN, self).__init__()
        self.tdnn1 = nn.Conv1d(in_channels=inchannel, out_channels=512, kernel_size=5, dilation=1)
        self.bn_tdnn1 = nn.BatchNorm1d(512, momentum=0.1, affine=False)
        self.dropout_tdnn1 = nn.Dropout(p=p_dropout)

        self.tdnn2 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=5, dilation=2)
        self.bn_tdnn2 = nn.BatchNorm1d(512, momentum=0.1, affine=False)
        self.dropout_tdnn2 = nn.Dropout(p=p_dropout)

        self.tdnn3 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=7, dilation=3)
        self.bn_tdnn3 = nn.BatchNorm1d(512, momentum=0.1, affine=False)
        self.dropout_tdnn3 = nn.Dropout(p=p_dropout)

        self.tdnn4 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=1, dilation=1)
        self.bn_tdnn4 = nn.BatchNorm1d(512, momentum=0.1, affine=False)
        self.dropout_tdnn4 = nn.Dropout(p=p_dropout)

        self.tdnn5 = nn.Conv1d(in_channels=512, out_channels=1500, kernel_size=1, dilation=1)
        self.bn_tdnn5 = nn.BatchNorm1d(1500, momentum=0.1, affine=False)
        self.dropout_tdnn5 = nn.Dropout(p=p_dropout)

        self.fc1 = nn.Linear(3000,512)
        self.fc1extra = nn.Linear(512,512)
        self.bn_fc1 = nn.BatchNorm1d(512, momentum=0.1, affine=False)
        self.dropout_fc1 = nn.Dropout(p=p_dropout)
        self.aap1 = nn.AdaptiveAvgPool1d(512)

        self.fc2 = nn.Linear(512,512)
        self.bn_fc2 = nn.BatchNorm1d(512, momentum=0.1, affine=False)
        self.dropout_fc2 = nn.Dropout(p=p_dropout)
        self.aap2 = nn.AdaptiveAvgPool1d(512)

        self.fc3 = nn.Linear(512,numSpkrs)

    def forward(self, x):
        # Note: x must be (batch_size, feat_dim, chunk_len)
        x = self.dropout_tdnn1(self.bn_tdnn1(F.relu(self.tdnn1(x))))
        x = self.dropout_tdnn2(self.bn_tdnn2(F.relu(self.tdnn2(x))))
        x = self.dropout_tdnn3(self.bn_tdnn3(F.relu(self.tdnn3(x))))
        x = self.dropout_tdnn4(self.bn_tdnn4(F.relu(self.tdnn4(x))))
        x = self.dropout_tdnn5(self.bn_tdnn5(F.relu(self.tdnn5(x))))
        
        if self.training:
            shape = x.size()
            noise = torch.cuda.FloatTensor(shape)
            torch.randn(shape, out=noise)
            
        stats = torch.cat((x.mean(dim=2), x.std(dim=2)), dim=1)
        #print(stats.shape)
        # x = self.dropout_fc1(self.bn_fc1(F.relu((self.fc1(stats)))))
        # x = self.dropout_fc2(self.bn_fc2(F.relu(self.fc2(x))))
        x = self.dropout_fc1(self.aap1(F.relu((self.fc1(stats)))))
        x = self.dropout_fc2(self.aap1(F.relu(self.fc2(x))))
        x = self.fc3(x)
        return x



class xvecExtraction(nn.Module):

    def __init__(self, numSpkrs, p_dropout, inchannel):
        super(xvecExtraction, self).__init__()
        self.tdnn1 = nn.Conv1d(in_channels=inchannel, out_channels=512, kernel_size=5, dilation=1)
        self.bn_tdnn1 = nn.BatchNorm1d(512, momentum=0.1, affine=False)

        self.tdnn2 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=5, dilation=2)
        self.bn_tdnn2 = nn.BatchNorm1d(512, momentum=0.1, affine=False)

        self.tdnn3 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=7, dilation=3)
        self.bn_tdnn3 = nn.BatchNorm1d(512, momentum=0.1, affine=False)

        self.tdnn4 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=1, dilation=1)
        self.bn_tdnn4 = nn.BatchNorm1d(512, momentum=0.1, affine=False)

        self.tdnn5 = nn.Conv1d(in_channels=512, out_channels=1500, kernel_size=1, dilation=1)
        self.bn_tdnn5 = nn.BatchNorm1d(1500, momentum=0.1, affine=False)

        self.fc1 = nn.Linear(3000,512)
        #self.aap1 = nn.AdaptiveAvgPool1d(512)

        self.fc2 = nn.Linear(512,512)


    def forward(self, x):
        x = torch.tensor(x, dtype=self.tdnn1.weight.dtype)
        x = self.bn_tdnn1(F.relu(self.tdnn1(x)))
        x = self.bn_tdnn2(F.relu(self.tdnn2(x)))
        x = self.bn_tdnn3(F.relu(self.tdnn3(x)))
        x = self.bn_tdnn4(F.relu(self.tdnn4(x)))
        x = self.bn_tdnn5(F.relu(self.tdnn5(x)))
            
        stats = torch.cat((x.mean(dim=2), x.std(dim=2)), dim=1)

        x = F.relu((self.fc1(stats)))

        return x