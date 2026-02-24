import torch
from ModifiedLeNet5_train import ModifiedLeNet5
from torchvision import transforms
import torch.nn.functional as F


# float32 tensor of shape (batch size, 1, 40, 40)
X_validation = torch.load('./affNIST Data Problem 2/X_validation')
# int64 tensor of shape (batch size)
y_validation = torch.load('./affNIST Data Problem 2/y_validation')

model = ModifiedLeNet5()
path = './LeNet5_2.pth'
model.load_state_dict(torch.load(path))

with torch.no_grad():
    outputs = model(X_validation)
    _, predicted = torch.max(outputs, 1)
    validation_error = predicted.ne(y_validation).sum() / float(y_validation.shape[0]) * 100
    
    print(f'Final Validaton Error {validation_error}%')
