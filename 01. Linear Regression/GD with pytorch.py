import torch

# 1: prepare dataset
x_data = torch.Tensor([[1.0], [2.0], [3.0]])
y_data = torch.Tensor([[2.0], [4.0], [6.0]])

# 2: design model
class LinearModel(torch.nn.Module):
    def __init__(self):
        super(LinearModel, self).__init__()
        self.linear = torch.nn.Linear(1, 1)  # Linear(w,b)
        
    def forward(self, x):
        y_hat = self.linear(x)
        return y_hat
        
model = LinearModel()

# 3: loss function & optimization method
criterion = torch.nn.MSELoss(size_average = True)  # size_average: whether /n
optimizer = torch.optim.SGD(model.parameters(), lr = 0.01)  # also with momentum

# 4: Training Circle
for epoch in range(100):
    y_hat = model(x_data)   # forward
    loss = criterion(y_hat, y_data)
    print(epoch, loss)

    optimizer.zero_grad()   # set grad=0 before backward
    loss.backward()
    optimizer.step()    #update

print('w =', model.linear.weight.item())  # use item cuz weight is a matrix
print('b =', model.linear.bias.item())

x_test = torch.Tensor([[4.0]])
y_test = model(x_test)
print('y_hat =', y_test.data)