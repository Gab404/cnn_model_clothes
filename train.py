import torch
from torchvision import datasets, transforms
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from cnn_net import MyNet

num_classes = 10
num_epochs = 2
learning_rate = 0.001
batch_size = 64

# Transform each image into tensor
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])

# Set the training loader
train_data = datasets.FashionMNIST('../data', train=True, download=True, transform=transform)
# Set the testing loader
test_data = datasets.FashionMNIST('../data', train=False, download=True, transform=transform)

label_dic = {
	0: "T-shirt/top",
	1: "Trouser",
	2 : "Pullover",
	3 : "Dress ",
	4 : "Coat ",
	5 : "Sandal ",
	6 : "Shirt ",
	7 : "Sneaker ",
	8 : "Bag ",
	9 : "Ankle boot"
}

# put my data to batch (64 images in each batch)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)

model = MyNet()

# loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

#training loop
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        #forward
        outputs = model(images)
        loss = criterion(outputs, labels)
        #backwards
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (i + 1) % 2000 == 0:
            print(f'epoch {epoch + 1} / {num_epochs}, loss = {loss.item():.4f}')
print("End of training")

with torch.no_grad():
    n_correct = 0
    n_samples = 0
    n_class_correct = [0 for i in range(10)]
    n_class_samples = [0 for i in range(10)]
    for images, labels in test_loader:
        outputs = model(images)
        
        _, predicted = torch.max(outputs.data, 1)
        n_samples += labels.size(0)
        n_correct += (predicted == labels).sum().item()

        for i in range(batch_size):
            label = labels[i]
            pred = predicted[i]
            if (label == pred):
                n_class_correct[label] += 1
            n_class_samples[label] += 1
            
    acc = 100.0 * n_correct / n_samples
    print(f'Accuracy of the network on the 10000 test images: {acc} %')

examples = iter(test_loader)
example_data, example_targets = examples.next()

output = model(example_data)
_, pred = torch.max(output.data, 1)
for i in range(batch_size):
    plt.subplot(2,3, i + 1)
    int_pred = int(pred[i])
    plt.title(label_dic[int_pred])
    plt.imshow(example_data[i][0], cmap='gray')
plt.show()