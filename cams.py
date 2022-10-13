from torchcam.methods import GradCAM, GradCAMpp, SmoothGradCAMpp

cam_dict = {
    'GradCAM': GradCAM,
    'GradCAMpp': GradCAMpp,
    'SmoothGradCAMpp': SmoothGradCAMpp,
}