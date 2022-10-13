import torch
from torchvision import transforms
import matplotlib.pyplot as plt
from torchvision.transforms.functional import to_pil_image
import numpy as np
from cams import cam_dict
from torchcam.utils import overlay_mask

from networks import resnet18
    
def plot_cam(map, image, name='default.png'):
    result = overlay_mask(to_pil_image(image), to_pil_image(map[0].squeeze(0), mode='F'), alpha=0.5)
    
    plt.imshow(result)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(name)
    plt.close()
        
def plot_cam_by_dataloader(model, dataloader, cam_type='GradCAM'):
    model.eval()
    cam_extractor = cam_dict[cam_type](model)
    
    for idx, (img, *kwarg) in enumerate(dataloader):
        out = model(img.unsqueeze(0))
        activation_map = cam_extractor(out.squeeze(0).argmax().item(), out)

        plot_cam(activation_map[0], img)    
    
if __name__ == '__main__':
    from torchvision.io.image import read_image
    
    model = resnet18(pretrained=True).eval()
    cam_extractor = cam_dict['GradCAMpp'](model)

    img = read_image('1.png')[:3] / 255.0

    transform = transforms.Compose([transforms.Resize((224, 224)),
                                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    img = transform(img)

    out = model(img.unsqueeze(0))
    activation_map = cam_extractor(out.squeeze(0).argmax().item(), out)

    plot_cam(activation_map[0], img)