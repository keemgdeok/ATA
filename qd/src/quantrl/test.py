import torch
cuda = True
device = torch.device("cuda" if cuda and torch.cuda.is_available() else "cpu")
if device == "cpu":
    print("Device is cpu")
    print(device)
    
else:
    print("Device is gpu")
    print(device)
    