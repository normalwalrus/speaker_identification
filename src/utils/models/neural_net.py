
import torch.nn as nn

class FeedForwardNN(nn.Module):

    def __init__(self,in_channel, output, dropout):
        super(FeedForwardNN, self).__init__()
        self.ln1 = nn.Linear(in_channel, 512)
        self.ln2 = nn.Linear(512, 512)
        self.ln3 = nn.Linear(512, output)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.ln1(x)
        x = self.dropout(x)
        x = self.ln2(x)
        x = self.dropout(x)
        x = self.ln3(x)

        return x

class SimpleNN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(SimpleNN, self).__init__()
        self.input_size = input_size
        self.l1 = nn.Linear(input_size, 512) 
        self.relu = nn.ReLU()
        self.leaky_relu = nn.LeakyReLU()
        self.l2 = nn.Linear(512, 1536)  
        self.l3 = nn.Linear(1536, 512)
        self.l4 = nn.Linear(512, num_classes)
        #self.softmax = nn.Softmax(dim=0)
    
    def forward(self, x):
        out = self.l1(x)
        out = self.leaky_relu(out)
        out = self.l2(out)
        out = self.leaky_relu(out)
        out = self.l3(out)
        out = self.leaky_relu(out)
        out = self.l4(out)
        #out = self.softmax(out)
        
        return out

class ConvNN(nn.Module):
    # ----------------------------
    # Build the model architecture
    # ----------------------------
    def __init__(self, output, dropout = 0.5):
        super().__init__()
        conv_layers = []

        # First Convolution Block with Relu and Batch Norm.
        self.conv1 = nn.Conv2d(1, 4, kernel_size=(2, 2), stride=(1, 1), padding=(1))
        self.relu1 = nn.LeakyReLU()
        self.bn1 = nn.BatchNorm2d(4)
        #init.kaiming_normal_(self.conv1.weight, a=0.1)
        self.conv1.bias.data.zero_()
        conv_layers += [self.conv1, self.relu1, self.bn1]

        # Second Convolution Block
        self.conv2 = nn.Conv2d(4, 16, kernel_size=(2, 2), stride=(1, 1), padding=(1))
        self.relu2 = nn.LeakyReLU()
        self.bn2 = nn.BatchNorm2d(16)
        #init.kaiming_normal_(self.conv2.weight, a=0.1)
        self.conv2.bias.data.zero_()
        conv_layers += [self.conv2, self.relu2, self.bn2]

        # Second Convolution Block
        self.conv3 = nn.Conv2d(16, 32, kernel_size=(2, 2), stride=(1, 1), padding=(1))
        self.relu3 = nn.LeakyReLU()
        self.bn3 = nn.BatchNorm2d(32)
        #init.kaiming_normal_(self.conv3.weight, a=0.1)
        self.conv3.bias.data.zero_()
        conv_layers += [self.conv3, self.relu3, self.bn3]

        # Second Convolution Block
        self.conv4 = nn.Conv2d(32, 64, kernel_size=(2, 2), stride=(1, 1), padding=(1))
        self.relu4 = nn.LeakyReLU()
        self.bn4 = nn.BatchNorm2d(64)
        #init.kaiming_normal_(self.conv4.weight, a=0.1)
        self.conv4.bias.data.zero_()
        conv_layers += [self.conv4, self.relu4, self.bn4]

        # Linear Classifier
        self.ap = nn.AdaptiveAvgPool2d(output_size=4)
        self.dropout = nn.Dropout(dropout)
        self.lin = nn.Linear(in_features=1024, out_features=output)

        # Wrap the Convolutional Blocks
        self.conv = nn.Sequential(*conv_layers)
 
    # ----------------------------
    # Forward pass computations
    # ----------------------------
    def forward(self, x):
        # Run the convolutional blocks
        x = self.conv(x)

        # Adaptive pool and flatten for input to linear layer
        x = self.ap(x)
        x = x.view(x.shape[0], -1)

        # Linear layer
        x = self.dropout(x)
        x = self.lin(x)

        # Final output
        return x
    