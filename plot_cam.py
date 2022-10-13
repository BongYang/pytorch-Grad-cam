import torch
from torch.utils import data
from torchvision import transforms
from torchvision import datasets
import matplotlib.pyplot as plt
import numpy as np
from cams import cam_dict

from networks import resnet18

def plot_cam():
    
def plot_cam():
    transform = transforms.Compose([transforms.Resize((224, 224)), 
                                    transforms.ToTensor()])
    
    model = resnet18().eval()

    img, _ = next(iter(dataloader))

    pred = model(img)
    
    
if __name__ == '__main__':
    plot_cam()