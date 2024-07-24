import torch

# Check if CUDA is available
print("CUDA available: ", torch.cuda.is_available())

# Check the CUDA version
print("CUDA version: ", torch.version.cuda)

# Check the cuDNN version
print("cuDNN version: ", torch.backends.cudnn.version())

# Check the number of available GPUs
print("Number of GPUs: ", torch.cuda.device_count())

# Create a simple tensor operation and move it to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
a = torch.tensor([1.0, 2.0], device=device)
b = torch.tensor([3.0, 4.0], device=device)
c = a + b

print("Result of tensor addition on GPU:\n", c)
