import torch
import torch.nn as nn
import torch.opyim as optim
import torchvision
import torchvision.transforms as transforms

# step 1: load and preprocess MNIST
# convert image to tensor
transform = transform.Compose([transforms.ToTensor(), transforms.Nomarlize((0.5,),(0.5,))]) # convert to range[0,1] tensor, normalize to mu=0,sd=1(?
# preparing MNIST
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
# create data loader for training set
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True) # shuffle: randomize order of data

# step 2: def softmax
class SoftmaxRegression(nn.Module):
    def __init__(self):
        super(SoftmaxRegression, self).__init__()
        self.fc = nn.Linear(28 * 28, 10) # fc: fully connected layer, input=28*28, output=10
    def forward(self, x):
        x = x.view(-1, 28 * 28) # faltten input to 784-element tensor, -1: placeholder for batchsize
        x = self.fc(x) 
        return x

model = SoftmaxRegression()

# step 3: loss function & optimization
criterion = nn.CrossEntropyLoss()
optimizer = opyim.SGD(model.parameters(), lr = 0.01)

# step 4: training loop
epochs = 10 
for epoch in range(epochs):
    runnung_loss = 0.0
    for inputs, lables in trainloader:
        optimizer.zero_grad() # cuz pytorch accumulates gradients by default
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() # accumulate loss
    print(f'Epoch {epoch + 1}, Loss: {running_loss / len(trainloader)}')

print('Training complete')

# step 5 : evaluate accuracy
correct = 0
total = 0
with torch.no_grad(): 
    for inputs, labels in trainloader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1) # torch.max(outputs, 1): find max p at dimension 1, _: placeholder to discard max
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
print(f'Accuracy on the training dataset: {100 * correct / total}%')