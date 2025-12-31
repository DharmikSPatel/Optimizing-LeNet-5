import pandas as pd
import torch
from torchvision import transforms
from PIL import Image
import io


# Download Data
splits = {'train': 'mnist/train-00000-of-00001.parquet', 'test': 'mnist/test-00000-of-00001.parquet'}
df_train = pd.read_parquet("hf://datasets/ylecun/mnist/" + splits["train"])
df_test = pd.read_parquet("hf://datasets/ylecun/mnist/" + splits["test"])

# Coveret each row which is 28x28 PIL image, [0, 1] to 32x32 Tensor [-.1, 1.175]
white, black = -.1, 1.175
std = 1/(black-white)
mean = std/10
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Pad(padding=2, fill=0),
     transforms.Normalize(mean, std)
    ]
)
def process_row(row):
    image_bytes = row['image']['bytes']
    pil_image = Image.open(io.BytesIO(image_bytes))
    tensor_image = transform(pil_image)
    return tensor_image, row['label']

processed_df_train = df_train.apply(process_row, axis=1)
processed_df_test = df_test.apply(process_row, axis=1)

# Make training tensors
X_train = torch.stack([item[0] for item in processed_df_train]) 
y_train = torch.tensor([item[1] for item in processed_df_train])
# Make testing tensors
X_test = torch.stack([item[0] for item in processed_df_test]) 
y_test = torch.tensor([item[1] for item in processed_df_test])

print("X_train shape", X_train.shape)
print("y_train shape", y_train.shape)
print("X_test shape",  X_test.shape)
print("y_test shape",  y_test.shape)

# save tensors
torch.save(X_train, './MNIST Data Problem 1/X_train')
torch.save(y_train, './MNIST Data Problem 1/y_train')
torch.save(X_test, './MNIST Data Problem 1/X_test')
torch.save(y_test, './MNIST Data Problem 1/y_test')