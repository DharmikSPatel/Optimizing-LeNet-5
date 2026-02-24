import pandas as pd
import torch, torchvision
from torchvision import transforms
from scipy.io import loadmat
from PIL import Image
import io

# Download Data
splits = {'train': 'mnist/train-00000-of-00001.parquet', 'test': 'mnist/test-00000-of-00001.parquet'}
df_train = pd.read_parquet("hf://datasets/ylecun/mnist/" + splits["train"])
validation_mat = loadmat('./affNIST Data Problem 2/affNISTvalidation.mat') # from online

# Data Augmentation - affine trasnfromations
'''
for each data sample, resize to 40x40, so add 6 pixel padding on all sides
then apply 10 random affine transformations per sample, yeilding 600,000 samples

final X_train shape = 600,000 x 1 x 40 x 40
final y_train shape = 600,000 class labels
'''
# Coveret each row which is 28x28 PIL image, [0, 1] to 40x40 affine Tensors
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Pad(padding=6, fill=0)
    ]
)
affine = transforms.RandomAffine(degrees=20, shear=.2, translate=(0.3, 0.3), scale=(0.7, 1.3))

def process_row(row):
    image_bytes = row['image']['bytes']
    pil_image = Image.open(io.BytesIO(image_bytes))
    tensor_image = transform(pil_image)
    label = row['label']
    
    imgs = [(tensor_image, label)]
    for _ in range(9):
        imgs.append((affine(tensor_image), label))
    return imgs

processed_df_train = df_train.apply(process_row, axis=1).explode(ignore_index=True)
# Make training tensors
X_train = torch.stack([item[0] for item in processed_df_train]) 
y_train = torch.tensor([item[1] for item in processed_df_train])

# # show data.
# grid = torchvision.utils.make_grid(X_train[0:100], nrow=10, padding=5, pad_value=1)
# toPIL = transforms.ToPILImage()
# grid = toPIL(grid)
# grid.save('P2_transformed_data_sample.png')
# grid.show()


# Used for validation, need to see if there is overfitting or not
X_validation = validation_mat['affNISTdata']['image'][0, 0]
y_validation = validation_mat['affNISTdata']['label_int'][0, 0]
validation_size = X_validation.shape[1]

# Make validation tensors
X_validation = (torch.unsqueeze(torch.stack([torch.tensor(X_validation[:, i]).reshape(40, 40) for i in range(validation_size)]), 1))/255
y_validation = torch.tensor(y_validation, dtype=torch.int64).reshape(validation_size)


print(X_train.shape, X_train.dtype)
print(y_train.shape, y_train.dtype)
print(X_validation.shape, X_validation.dtype)
print(y_validation.shape, y_validation.dtype)

# save tensors
torch.save(X_train, './affNIST Data Problem 2/X_train')
torch.save(y_train, './affNIST Data Problem 2/y_train')
torch.save(X_validation, './affNIST Data Problem 2/X_validation')
torch.save(y_validation, './affNIST Data Problem 2/y_validation')