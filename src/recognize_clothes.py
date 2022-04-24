import torch
from torchvision import datasets, transforms
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import random
from cnn_net import MyNet

FILE = "save/my_model.txt"
batch_size = 4

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

# Transform each image into tensor
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])

# Set the training loader
train_data = datasets.FashionMNIST('../data', train=True, download=True, transform=transform)
# Set the testing loader
test_data = datasets.FashionMNIST('../data', train=False, download=True, transform=transform)

# put my data to batch (4 images in each batch)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)

#load model
loaded_model = MyNet()
loaded_model.load_state_dict(torch.load(FILE))
loaded_model.eval()

examples = iter(test_loader)
for i in range(random.randint(2, 500)):
    example_data, example_targets = examples.next()

output = loaded_model(example_data)
_, pred = torch.max(output.data, 1)
for i in range(batch_size):
    plt.subplot(2,3, i + 1)
    int_pred = int(pred[i])
    plt.title(label_dic[int_pred])
    plt.imshow(example_data[i][0], cmap='gray')
plt.show()