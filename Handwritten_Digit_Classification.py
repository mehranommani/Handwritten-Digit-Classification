import time
import os
import random
import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms as T
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from tempfile import TemporaryDirectory
from PIL import Image


transform = T.Compose([
    T.ToTensor(),
    T.Resize((28, 28))
])

ds = torchvision.datasets.ImageFolder(root="/content/drive/MyDrive/images/train", transform=transform)
ds_test = torchvision.datasets.ImageFolder(root="/content/drive/MyDrive/images/test", transform=transform)

"""##sklearn Classification"""

# Preprocess the data
X_train = []
y_train = []
for image, label in ds:
    image_array = image.numpy()
    X_train.append(image_array)
    y_train.append(label)

X_test = []
y_test = []
for image, label in ds_test:
    image_array = image.numpy()
    X_test.append(image_array)
    y_test.append(label)

# Flatten the image data
X_train_flat = [image.flatten() for image in X_train]
X_test_flat = [image.flatten() for image in X_test]

# Convert lists to NumPy arrays
X_train_flat = np.array(X_train_flat)
X_test_flat = np.array(X_test_flat)

# Create and train the classifier
clf = SVC()
clf.fit(X_train_flat, y_train)

# Evaluate the classifier
accuracy = clf.score(X_test_flat, y_test)
print("Accuracy:", accuracy)

"""##CNN"""

dataset_sizes = {
    'train': len(ds),
    'val': len(ds_test)
}

idx_to_class = {v: int(k) for (k, v) in ds.class_to_idx.items()}

dataloaders = {
    "train": torch.utils.data.DataLoader(ds, batch_size=4, shuffle=True, num_workers=4),
    "val": torch.utils.data.DataLoader(ds_test, batch_size=4, shuffle=True, num_workers=4),
}


def train_model(model, criterion, optimizer, scheduler, dataset_sizes, num_epochs=25):
    since = time.time()

    best_model_params_path = os.path.join('/content/', 'best_model_params.pt')

    torch.save(model.state_dict(), best_model_params_path)
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to("cuda")
                labels = labels.to("cuda")

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                torch.save(model.state_dict(), best_model_params_path)

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    # load best model weights
    model.load_state_dict(torch.load(best_model_params_path))
    return model

class_names = ds.classes


# Define the path to the "train" directory
train_dir = '/content/drive/MyDrive/images/train'

# List subdirectories (labels)
subdirectories = [f for f in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, f))]

# Randomly select a label (subdirectory)
random_label = random.choice(subdirectories)

# List image files within the selected subdirectory
image_dir = os.path.join(train_dir, random_label)
image_files = [f for f in os.listdir(image_dir) if f.endswith('.png')]

# Randomly select an image file
random_index = random.randint(0, len(image_files) - 1)
random_image_file = image_files[random_index]

# Load the randomly selected image
image_path = os.path.join(image_dir, random_image_file)
image = Image.open(image_path)

# Define a transformation
transform = T.Compose([
    T.Resize((224, 224), antialias=True),  # Set antialias=True to suppress the warning
    T.ToTensor(),
    ])

# Apply the transformation
image = transform(image)

# Display the image and its label
plt.imshow(image.permute(1, 2, 0), cmap='gray')  # PyTorch tensors are in (C, H, W) format, so we permute it to (H, W, C)
print(f'index: {random_label}')
plt.show()

model_ft = torchvision.models.resnet18(weights='IMAGENET1K_V1')
num_ftrs = model_ft.fc.in_features
# Here the size of each output sample is set to 2.
# Alternatively, it can be generalized to ``nn.Linear(num_ftrs, len(class_names))``.
model_ft.fc = torch.nn.Linear(num_ftrs, len(class_names))

model_ft = model_ft.to("cuda")

criterion = torch.nn.CrossEntropyLoss()

# Observe that all parameters are being optimized
optimizer_ft = torch.optim.SGD(model_ft.parameters(), lr=0.0001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, dataset_sizes,
                       num_epochs=10)

torch.save(model_ft.state_dict(), '/content/drive/MyDrive/mehranommani_classifier.pth')
