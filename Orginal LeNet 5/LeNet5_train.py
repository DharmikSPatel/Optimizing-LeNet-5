import torch
from torchsummary import summary
import torch.utils.data as torchutils
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
import numpy as np

'''
Notes from LeNet5 Paper
CANCEL Learning Rate Sceduler
- .0005 x2 iter
- .0002 x3 iter
- .0001 x3 iter
- .00005 x4 iter
- .00001 x8 iter

DONE Custum Loss Function
- Equation 9
- j = .1 and i denotes incrrect classs

DONE Input
- 32x32 image
- -.1 is white, 1.175 is black

DONE Initalization of weights and bias
- Weigths and biases are rand bwn [-2.4/Fi, 2.4/Fi]. Fi is the number of inputs in the layer

DONE Each layer up to F6 has a tanh activation function
- f(a) = Atanh(Sa)
- A = 1.7159
- S = 2/3
- f(a) = Atanh(Sa)

- DONE Epoch 20
- DONE Batch size = 1

DONE Optimzer
- DONE SGD. 
- CANCEL use the exact second derivative

Layers
- DONE custum pooling layers
- DONE selective C3 Layer
- DONE RBF
'''


class SubSamplingLayer(nn.Module):
    def __init__(self, in_channels, kernel_size, stride):
        super(SubSamplingLayer, self).__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.stride = stride

        self.weight = nn.Parameter(torch.zeros(1, self.in_channels, 1, 1))
        nn.init.uniform_(self.weight, -2.4/self.in_channels, 2.4/self.in_channels)
        
        self.bias = nn.Parameter(torch.zeros(1, self.in_channels, 1, 1))
        nn.init.uniform_(self.bias, -2.4/self.in_channels, 2.4/self.in_channels)
    def forward(self, x):
        # X is a image of size (-1, in_channel, H, W)

        # need to: (sum each 2x2 neighbor hood)*weigth+bias PER channel

        # this will help us sum each 2x2 block. 
        # I used avgpool which is sum/len so I just mutiplied by the len to just get the sum
        x = F.avg_pool2d(x, self.kernel_size, self.stride) * self.kernel_size*2
        
        # muitple weight and add bias
        x = x*self.weight + self.bias

        return x



class RBFLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = torch.load('./RBF Kernel/FinalKernels/rbf_weights')
        self.weights.requires_grad = False
        # print(self.weights.shape)
    def forward(self, x):
        # print(x.shape)
        new_xs = []
        for item in x:
            # print("item", item.shape)
            new_x = ((item-self.weights)**2).sum(1)
            # print("new_x", new_x, new_x.shape)
            new_xs.append(new_x)
        x = torch.stack(new_xs)
        # print(x.shape)
        return x
class SelectiveC3(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # in a normal Conv2dlayer, the weights are ([16, 6, 5, 5])
        # change them to ([10, 6, 5, 5])
        # bc each input channel goes to only certain 10/16 output channels
        self.weight = nn.Parameter(torch.zeros(10, 6, 5, 5))
        nn.init.uniform_(self.weight, -2.4/self.in_channels, 2.4/self.in_channels)
        
        # need to add to each one of the output layers
        self.bias = nn.Parameter(torch.zeros(1, 16, 1, 1))
        nn.init.uniform_(self.bias, -2.4/self.in_channels, 2.4/self.in_channels)
        
        # list of each of the 10 output channels each 6 input features are connected to
        self.connections = [[0, 4, 5, 6, 9, 10, 11, 12, 14, 15],
                            [0, 1, 5, 6, 7, 10, 11, 12, 13, 15],
                            [0, 1, 2, 6, 7, 8, 11, 13, 14, 15],
                            [1, 2, 3, 6, 7, 8, 9, 12, 14, 15],
                            [2, 3, 4, 7, 8, 9, 10, 12, 13, 15],
                            [3, 4, 5, 8, 9, 10, 11, 13, 14, 15]]
    def forward(self, x):
        # input is -1 x 6 x 14 x 14
        '''
        Reuqired format for 
        F.conv2d(input, weight)
        input: image, (batch, in_channel, H, W)
        weights:      (out_channel, in_channel, kH, kW)
        bias:         (out_channel)
        '''

        final = torch.zeros(x.size(0), self.out_channels, 10, 10)
        for input_channel in range(self.in_channels):
            # the unsqueeze keeps in the required format mentioned above
            # size of [batch, 1, 14, 14] 
            x_data = x[:, input_channel, :, :].unsqueeze(1)
            # still of size [10, 1, 5, 5]
            input_channel_weights = self.weight[:, input_channel, :, :].unsqueeze(1)
            
            final[:, self.connections[input_channel], :, :] += F.conv2d(x_data, input_channel_weights) + self.bias[:,self.connections[input_channel],:,:]
        

        return final  

        

class Tanh(nn.Module):
    def forward(self, x):
        return 1.7159*torch.tanh(x*2/3)
def custumLoss(y_pred, y):
    # y_pred is a tensor of size [-1, 10]. it is the RBF output
    # y is a tensor of size [-1, 10]. it is one-hot-encoded class labels
    j = .1
    return (y_pred[y == 1] + torch.log(np.exp(-j)+torch.exp(-y_pred[y == 0]).sum())).mean()

class LeNet5(nn.Module):
    def __init__(self):
        super().__init__()
        self.C1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=0)
        self.init_weights_biases(self.C1, self.C1.in_channels)
        
        self.S2 = SubSamplingLayer(in_channels=6, kernel_size=2, stride=2)
        
        self.C3 = SelectiveC3(in_channels=6, out_channels=16)

        self.S4 = SubSamplingLayer(in_channels=16, kernel_size=2, stride=2)
        
        self.C5 = nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5, stride=1, padding=0)
        self.init_weights_biases(self.C5, self.C5.in_channels)
        
        self.F6 = nn.Linear(in_features=120, out_features=84)
        self.init_weights_biases(self.F6, self.F6.in_features)
        self.RBF = RBFLayer()

    def init_weights_biases(self, layer, layer_in: int, c = 2.4):
        nn.init.uniform_(layer.weight, -c/layer_in, c/layer_in)
        nn.init.uniform_(layer.bias, -c/layer_in, c/layer_in)
    
    def forward(self, x):
        tanh = Tanh()
        x = tanh(self.C1(x))
        x = tanh(self.S2(x))
        x = tanh(self.C3(x))
        x = tanh(self.S4(x))
        x = tanh(self.C5(x))
        x = torch.flatten(x, 1)
        x = tanh(self.F6(x))
        x = self.RBF(x)
        return x
        

if __name__ == '__main__':

    
    # Load Data from files. (used data.py)
    X_train = torch.load('./MNIST Data Problem 1/X_train')
    y_train = torch.load('./MNIST Data Problem 1/y_train')

    X_test = torch.load('./MNIST Data Problem 1/X_test')
    y_test = torch.load('./MNIST Data Problem 1/y_test')

    train_dataset = torchutils.TensorDataset(X_train, y_train)
    train_loader = torchutils.DataLoader(train_dataset, batch_size=1, shuffle=False)

    # Setup Model Architecure
    model = LeNet5()
    summary(model,(1,32,32))

    # optimizer
    base_lr = .0001
    optimizer = torch.optim.SGD(model.parameters(), lr=base_lr)
   
    num_epochs = 20

    # percentages
    train_errors = []
    test_errors = []

    # training loop
    n_total_batches = len(train_loader)
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            y_predicted = model(images) # forward prop
            one_hot_labels = F.one_hot(labels, num_classes=10)
            loss = custumLoss(y_predicted, one_hot_labels) 

            optimizer.zero_grad() # zero out gradients
            loss.backward() # calculate gradients
            optimizer.step() # update gradients

            if (i+1) % 2000 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Batch [{i+1}/{n_total_batches}], LR {base_lr}, Loss: {loss.item():.4f}')

        with torch.no_grad():
            outputs = model(X_train)
            _, predicted = torch.min(outputs, 1)
            train_error = predicted.ne(y_train).sum() / float(y_train.shape[0]) * 100
            train_errors.append(train_error.item())

            outputs = model(X_test)
            _, predicted = torch.min(outputs, 1)
            test_error = predicted.ne(y_test).sum() / float(y_test.shape[0]) * 100
            test_errors.append(test_error.item())
            
            print(f'Epoch [{epoch+1}/{num_epochs}], Train Error {train_error}%, Test Error {test_error}%')

    print("================================================================")
    path = './LeNet5_1.pth'
    torch.save(model.state_dict(), path)
    print('Finished Training & Saving Model to', path)
    print(f'Final Traning Error: {train_errors[-1]}%')
    print(f'Final Test Error: {test_errors[-1]}%')
    print("================================================================")

    # training/test error graphs
    plt.figure(figsize=(8, 6))
    epochs_list = np.arange(start=0, stop=num_epochs, step=1)
    plt.xticks(epochs_list)
    plt.plot(epochs_list, train_errors, marker='o', label='Training Error %')
    plt.plot(epochs_list, test_errors, marker='s', label='Test Error %')
    plt.xlabel('Epochs')
    plt.ylabel('Error Rate (%)')
    plt.legend()
    plt.savefig('test_train_error_perc.png')
    plt.show()


        