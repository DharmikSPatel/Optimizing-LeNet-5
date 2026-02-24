import torch
import torch.nn as nn
from scipy.io import loadmat
import torch.utils.data as torchutils
from torchsummary import summary
from torchvision import transforms
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
# Question 2

class ModifiedLeNet5(nn.Module):
    def __init__(self):
        super().__init__()
        self.C1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=0)
        self.S2 = nn.MaxPool2d(kernel_size=2)
        self.C3 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0)
        self.S4 = nn.MaxPool2d(kernel_size=2)
        self.C5 = nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5, stride=1, padding=0)
        self.S6 = nn.MaxPool2d(kernel_size=2)
        self.F7 = nn.Linear(in_features=120, out_features=84)
        self.F8 = nn.Linear(in_features=84, out_features=10)

    def forward(self, x):
        x = F.leaky_relu(self.C1(x))
        x = self.S2(x)
        x = F.leaky_relu(self.C3(x))
        x = self.S4(x)
        x = F.leaky_relu(self.C5(x))
        x = self.S6(x)
        x = torch.flatten(x, 1)
        x = F.leaky_relu(self.F7(x))
        x = self.F8(x)
        return x

if __name__ == "__main__":
    # Hyper-parameters 
    num_epochs = 20
    batch_size = 4
    learning_rate = 0.001

    # load data
    X_train = torch.load('./affNIST Data Problem 2/X_train')
    y_train = torch.load('./affNIST Data Problem 2/y_train')
    X_validation = torch.load('./affNIST Data Problem 2/X_validation')
    y_validation = torch.load('./affNIST Data Problem 2/y_validation')

    print(X_train.shape, X_train.dtype)
    print(y_train.shape, y_train.dtype)
    print(X_validation.shape, X_validation.dtype)
    print(y_validation.shape, y_validation.dtype)

    train_dataset = torchutils.TensorDataset(X_train, y_train)
    train_loader = torchutils.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # sample data. used to calculate train and val errors after each epoch. 
    # dont want to use whole data set bc it takes too long.
    # not used to train
    num = 50000
    sample_i_train = torch.randperm(X_train.size(0))[0:num]
    sample_X_train = X_train[sample_i_train]
    sample_y_train = y_train[sample_i_train]
    sample_i_validation = torch.randperm(X_validation.size(0))[0:num]
    sample_X_validation = X_validation[sample_i_validation]
    sample_y_validation = y_validation[sample_i_validation]

    model = ModifiedLeNet5()
    summary(model,(1,40,40))
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    # percentages
    train_errors = []
    validation_errors = []

    # training loop
    n_total_batches = len(train_loader)
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            y_predicted = model(images) # forward prop
            loss = criterion(y_predicted, labels)

            optimizer.zero_grad() # zero out gradients
            loss.backward() # calculate gradients
            optimizer.step() # update gradients

            if (i+1) % 2000 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Batch [{i+1}/{n_total_batches}], Loss: {loss.item():.4f}')

        path = f'./Problem 2 Models/ModifiedLeNet5_Epoch_{epoch+1}.pth'
        print('Finished Saving Model to', path)
        torch.save(model.state_dict(), path)

        with torch.no_grad():
            outputs = model(sample_X_train)
            _, predicted = torch.max(outputs, 1)
            train_error = predicted.ne(sample_y_train).sum() / float(sample_y_train.shape[0]) * 100
            train_errors.append(train_error.item())

            outputs = model(sample_X_validation)
            _, predicted = torch.max(outputs, 1)
            validation_error = predicted.ne(sample_y_validation).sum() / float(sample_y_validation.shape[0]) * 100
            validation_errors.append(validation_error.item())
            
            print(f'Epoch [{epoch+1}/{num_epochs}], Train Error {train_error}%, validation Error {validation_error}%')

    print("================================================================")
    print('Finished Training')
    print('Please pick the model with the shows the smallest overfitting')
    print("================================================================")
    
    # training/test error graphs
    plt.figure(figsize=(8, 6))
    epochs_list = np.arange(start=0, stop=num_epochs, step=1)
    plt.xticks(epochs_list)
    plt.plot(epochs_list, train_errors, marker='o', label='Training Error %')
    plt.plot(epochs_list, validation_errors, marker='s', label='Validation Error %')
    plt.xlabel('Epochs')
    plt.ylabel('Error Rate (%)')
    plt.legend()
    plt.savefig('P2_test_validation_error_perc.png')
    plt.show()