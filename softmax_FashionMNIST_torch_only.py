import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

# download Fashion MNIST
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]) # convert image to tensor & normalize 
trainset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
testset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform) 
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False) # shuffle: randomize the order of data

# def NN
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128) # fc1: fully-connected layer
        self.fc2 = nn.Linear(128, 10)  # 10 output classes
    def forward(self, x):
        x = x.view(-1, 28 * 28)  # Flatten the input to have a shape (batch_size, 28 * 28)
        x = torch.relu(self.fc1(x)) # use relu as the activation function between hidden layer
        x = self.fc2(x)
        return torch.softmax(x, dim=1) # use softmax for the output layer
net = Net()

# loss function and optimization method
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9) # this shit gpt3.5 even give me momentum!

# training loop
for epoch in range(10):  # the shit gpt give me epoch = only 5 ???
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        optimizer.zero_grad() # cuz pytorch acculumate gradient in bp by defaults
        outputs = net(inputs) # forward
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {running_loss/len(trainloader)}")

# testing loop
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images) # forward iterate
        _, predicted = torch.max(outputs.data, 1) # find the class of max probability
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Accuracy on the test dataset: {100 * correct / total}%")
