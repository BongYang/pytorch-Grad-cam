import torch
from torchvision import transforms
import matplotlib.pyplot as plt
from torchvision.transforms.functional import to_pil_image
import numpy as np
from cams import cam_dict
from torchcam.utils import overlay_mask

from networks.resnet_big import resnet18
    
def save_fig(target, name='default.png'):
    if isinstance(target, torch.Tensor):
        target = target.numpy().transpose(1, 2, 0)
        
    plt.imshow(target)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(name)
    plt.close()
        
def plot_cam_by_dataloader(model, dataloader, cam_type='GradCAM'):
    model.eval()
    cam_extractor = cam_dict[cam_type](model)
    
    for idx, (img, *kwarg) in enumerate(dataloader):
        out = model(img)
        activation_map = cam_extractor(out.squeeze(0).argmax().item(), out)

        result = overlay_mask(to_pil_image(img), to_pil_image(activation_map[0].squeeze(0), mode='F'), alpha=0.5)
    
        save_fig(result)

def plot_cam(model, img, cam_type='GradCAMpp'):
    model.eval()
    cam_extractor = cam_dict[cam_type](model)
    
    out = model(img.unsqueeze(0))
    activation_map = cam_extractor(out.squeeze(0).argmax().item(), out)

    result = overlay_mask(to_pil_image(img), to_pil_image(activation_map[0].squeeze(0), mode='F'), alpha=0.5)
    
    save_fig(result)
        
if __name__ == '__main__':
    from torchvision.io.image import read_image
    
    model = resnet18().eval()

    img = read_image('1.png')[:3] / 255.0
    transform = transforms.Compose([transforms.Resize((224, 224)),
                                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    img = transform(img)

    plot_cam(model, img)