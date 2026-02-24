import torch
from LeNet5_train import LeNet5, custumLoss
from torchvision import transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torcheval.metrics.functional import multiclass_confusion_matrix
import seaborn as sns

X_train = torch.load('./MNIST Data Problem 1/X_train')
y_train = torch.load('./MNIST Data Problem 1/y_train')

X_test = torch.load('./MNIST Data Problem 1/X_test')
y_test = torch.load('./MNIST Data Problem 1/y_test')

model = LeNet5()
model.load_state_dict(torch.load('./LeNet5_1.pth'))

with torch.no_grad():
    # setup transform to unnromalize so its btwn 0, 1 and convert to PIL
    # used to view images only
    
    # train and test errors at epoch 20 (ie once model is tranined)
    outputs = model(X_train)
    _, predicted = torch.min(outputs, 1)
    train_error = predicted.ne(y_train).sum() / float(y_train.shape[0]) * 100

    outputs = model(X_test)
    _, predicted = torch.min(outputs, 1)
    one_hot_labels = F.one_hot(y_test, num_classes=10)
    test_error = predicted.ne(y_test).sum() / float(y_test.shape[0]) * 100
    
    print(f'Epoch [{20}/{20}], Train Error {train_error}%, Test Error {test_error}%')

    
    # Most confusing sample
    unnormalize = transforms.Compose([transforms.Normalize(-.1, 1.275)])
    losses = torch.tensor([custumLoss(sample_pred, truth_label).item() for sample_pred, truth_label in zip(outputs, one_hot_labels)])
    most_confusing_images, captions = [], []
    for digit in range(10):
        # keeps teh losses for the digit, turns everything else to -inf
        class_losses = torch.where(predicted == digit, losses, -torch.inf)
        most_confusing_i = torch.argmax(class_losses)
        image = unnormalize(X_test[most_confusing_i]).permute(1, 2, 0).numpy()
        most_confusing_images.append(image)
        captions.append(f"Missclasfied as {predicted[most_confusing_i]} but truth label is {y_test[most_confusing_i]}")


    # confusion matrix
    # actual classes are the rows and the predicted classes are the cols
    cm = multiclass_confusion_matrix(-outputs, y_test, num_classes=10)
    print(cm)
    fig0, axes0 = plt.subplots(1,1, figsize=(8, 8))
    axes0.set_aspect('equal')
    sns.heatmap(cm, annot=True, fmt='g')
    plt.xlabel('Predicted Class')
    plt.ylabel('Actual Class')
    plt.savefig('cofusion_matrix.png')
    
    # Show most confusing samples
    fig, axes = plt.subplots(2, 5, figsize=(20, 5))
    axes = axes.flatten()
    for ax, img, caption in zip(axes, most_confusing_images, captions):
        ax.imshow(img, cmap='gray')                     
        ax.axis('off')                     
        ax.set_title(caption, fontsize=10) 
    plt.tight_layout()
    plt.savefig("most_confusing_test_samples.png")  # Save the grid
    plt.show()


    
    