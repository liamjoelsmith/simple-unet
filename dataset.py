from imports import *
import os

"""
A custom image loader dataset
"""
class ImageLoader(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        # self.total = os.listdir(img_dir)
        # self.total.sort()

        self.total_images = os.listdir(os.path.join(self.img_dir, "images"))
        self.total_images.sort()
        self.total_labels = os.listdir(os.path.join(self.img_dir, "labels"))
        self.total_labels.sort()

        self.transform = transform

    def __len__(self):
        return len(self.total_images)

    def __getitem__(self, index):
        transform = torchvision.transforms.ToTensor()
        image = os.path.join(self.img_dir, "images", self.total_images[index])
        label = os.path.join(self.img_dir, "labels", self.total_labels[index])
        image = transform(Image.open(image))
        label = transform(Image.open(label))
        if self.transform:
            image = self.transform(image)
            label = self.transform(label)
        return image, label

"""
Load in image(s) and transform data via custom ImageLoader, and create our DataLoader
"""
def load_train_dataset(batch_size=8, image_resize=256, relative_path="ISBI_dataset\\"):
    transform = Compose([
                    ToPILImage(),
                    Grayscale(),  # ensure images only have one channel
                    Resize(image_resize),  # ensure all images have same size
                    CenterCrop(image_resize),
                    ToTensor(),
                ])

    # load data, with the above transform applied
    train_imgs = ImageLoader(relative_path + "train",
                                transform=transform)
    return DataLoader(train_imgs, batch_size=batch_size, shuffle=False, num_workers=1)

"""
Load in image(s) and transform data via custom ImageLoader, and create our DataLoader
"""
def load_validation_dataset(batch_size=8, image_resize=256, relative_path="ISBI_dataset\\"):
    transform = Compose([
                    ToPILImage(),
                    Grayscale(),  # ensure images only have one channel
                    Resize(image_resize),  # ensure all images have same size
                    CenterCrop(image_resize),
                    ToTensor(),
                ])

    # load data, with the above transform applied
    val_imgs = ImageLoader(relative_path + "validation",
                                transform=transform)
    return DataLoader(val_imgs, batch_size=batch_size, shuffle=False, num_workers=1)

"""
A custom image loader dataset, specifically to load in the test dataset (only contains images, no labels)
"""
class TestLoader(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.img_dir = img_dir

        self.total_images = os.listdir(os.path.join(self.img_dir, "images"))
        self.total_images.sort()

        self.transform = transform

    def __len__(self):
        return len(self.total_images)

    def __getitem__(self, index):
        transform = torchvision.transforms.ToTensor()
        image = os.path.join(self.img_dir, "images", self.total_images[index])
        image = transform(Image.open(image))
        if self.transform:
            image = self.transform(image)
        return image

"""
Load in image(s) and transform data via custom TestLoader, and create our DataLoader
"""
def load_test_dataset(batch_size=8, image_resize=256, relative_path="ISBI_dataset\\"):
    transform = Compose([
                    ToPILImage(),
                    Grayscale(),  # ensure images only have one channel
                    Resize(image_resize),  # ensure all images have same size
                    CenterCrop(image_resize),
                    ToTensor(),
                ])

    # load data, with the above transform applied
    ims = TestLoader(relative_path + "test", transform=transform)
    return DataLoader(ims, batch_size=batch_size, shuffle=False, num_workers=1)