import torch, torchvision
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

from PIL import Image
import os
'''
Goal it to take the images in the ./Data folder
and then convert them to 7x12 kernels, and turn into
CNN layer.

export the wegihts as a singler tensor of size (10, 84)
'''

class ClampTransform(torch.nn.Module):
    def __init__(self, threshold=.5):
        super().__init__()
        self.threshold = threshold
    def forward(self, img):
        img = torch.where(img < self.threshold, 0., 1.)
        return img
# export kernels as jpgs and weight tensor
def generate_kernels():
    final = []
    for digit in range(10):
        digit_folder = f'./Data/{digit}'
        image_tensors = []
        transform_image_to_tensor = transforms.Compose([transforms.ToTensor()])
        for filename in os.listdir(digit_folder):
            if filename.endswith(('.jpeg')):  # Add image formats as needed
                image_path = os.path.join(digit_folder, filename)
                image = Image.open(image_path).convert('L')
                image_tensor = transform_image_to_tensor(image)
                image_tensors.append(image_tensor)
        image_tensors = torch.stack(image_tensors)
        # print(image_tensors.shape)

        average_image_tensor = image_tensors.mean(dim=0)
        # Clamp values: greater than 0.5 -> 1, less than or equal to 0.5 -> 0
        transform_tensor = transforms.Compose([transforms.CenterCrop((100, 80)),
                                                        transforms.Resize((12, 7),  InterpolationMode.BILINEAR), 
                                                        ClampTransform(threshold=.5)])
        final_tensor = transform_tensor(average_image_tensor)
        final.append(final_tensor)
        toPIL = transforms.ToPILImage()
        image = toPIL(final_tensor)
        image.save(f"./FinalKernels/{digit}.png")
    final = torch.stack(final)
    final = torch.where(final == 0, -1., 1.)
    # print(final)
    # conver to a 10x84 tensor. 84 weights per digit
    final = final.view(10, -1)
    torch.save(final, f"./FinalKernels/rbf_weights")
    print("final", final.shape)
def show_avgs():
    avgs = []
    for digit in range(10):
        digit_folder = f'./Data/{digit}'
        image_tensors = []
        transform_image_to_tensor = transforms.Compose([transforms.ToTensor()])
        for filename in os.listdir(digit_folder):
            if filename.endswith(('.jpeg')):  # Add image formats as needed
                image_path = os.path.join(digit_folder, filename)
                image = Image.open(image_path)
                image_tensor = transform_image_to_tensor(image)
                image_tensors.append(image_tensor)
        image_tensors = torch.stack(image_tensors)
        print(image_tensors.shape)

        average_image_tensor = image_tensors.mean(dim=0)
        centercrop = transforms.CenterCrop((100, 80))
        avgs.append(centercrop(average_image_tensor))
    grid = torchvision.utils.make_grid(avgs, nrow=5)
    toPIL = transforms.ToPILImage()
    grid = toPIL(grid)
    grid.show()   
def test_kernel(digit=9):
    digit_folder = f'./Data/{digit}'
    image_tensors = []
    transform_image_to_tensor = transforms.Compose([transforms.ToTensor()])
    for filename in os.listdir(digit_folder):
        if filename.endswith(('.jpeg')):  # Add image formats as needed
            image_path = os.path.join(digit_folder, filename)
            image = Image.open(image_path)
            image_tensor = transform_image_to_tensor(image)
            image_tensors.append(image_tensor)
    image_tensors = torch.stack(image_tensors)
    print(image_tensors.shape)

    average_image_tensor = image_tensors.mean(dim=0)

    # now test the transofrms
    images = []
    modes = [InterpolationMode.NEAREST, InterpolationMode.NEAREST_EXACT, InterpolationMode.BILINEAR, InterpolationMode.BICUBIC]
    for mode in modes:
        for t in range(5, 11):
            transform_tensor_to_image = transforms.Compose([transforms.CenterCrop((100, 80)), 
                                                            transforms.Resize((12, 7), interpolation=mode), 
                                                            ClampTransform(threshold=t/10)])
            image = transform_tensor_to_image(average_image_tensor)
            images.append(image)
    grid = torchvision.utils.make_grid(images, nrow=6)
    toPIL = transforms.ToPILImage()
    grid = toPIL(grid)
    grid.show()

if __name__ == '__main__':
    generate_kernels() # comment in this line if you want to regenerate the weights

    # helper methods

    # show_avgs()
    # for d in range(10):
    #     test_kernel(d)
    pass
