import torch


# FGM attack code
def fgm_attack(image, epsilon, data_grad):
    # Collect the element-wise sign of the data gradient
    grad_norms = torch.norm(data_grad)
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon * data_grad / grad_norms
    # Adding clipping to maintain [0,1] range
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    # Return the perturbed image
    return perturbed_image