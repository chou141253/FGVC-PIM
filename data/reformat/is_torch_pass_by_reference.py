import torch

"""
torch tensor pass by reference
"""

def add(b):
    b[0] += 100

if __name__ == "__main__":
    a = torch.ones([100])
    add(a)
    print(a)