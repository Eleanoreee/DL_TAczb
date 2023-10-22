#%%
# the number of parameter of the resnet

# 1. Define a function to count the number of parameters

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

