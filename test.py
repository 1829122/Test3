import torch

checkpoint = torch.load("epoch-184.pt", map_location = torch.device("cuda:3"))

print(checkpoint.keys())



