from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.utils.data as torchutils
import mnist
import torch
import numpy as np
from LeNet5_train import LeNet5
from torchvision import transforms



def test(dataloader: mnist.MNIST,model):
    #please implement your test code#                                                                                                                                                                      

    ###########################       
    total = len(dataloader)
    correct = 0
    for i, (image, label) in enumerate(dataloader):   
        output = model(image)
        _, predicted = torch.min(output, 1)
        assert(predicted.eq(label) == (predicted == label))
        if predicted.eq(label):
            correct += 1
    
    test_accuracy = correct /total * 100
    print("test accuracy:", test_accuracy)

def main():

    white, black = -.1, 1.175
    std = 1/(black-white)
    mean = std/10
    transform = transforms.Compose(
        [lambda x: x/255,
        transforms.Pad(padding=2, fill=0),
        transforms.Normalize(mean, std)
        ]
    )
    mnist_test=mnist.MNIST(split="test",transform=transform)
    print(mnist_test.__getitem__(1)[0].shape)
    # BATCH SIZE MUST BE 1 for test code to work
    test_dataloader= DataLoader(mnist_test,batch_size=1,shuffle=False)
    
    model = LeNet5()
    model.load_state_dict(torch.load('./LeNet5_1.pth'))
    
    test(test_dataloader,model)

if __name__=="__main__":
    main()