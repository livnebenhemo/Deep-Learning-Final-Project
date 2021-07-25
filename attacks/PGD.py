import torch


# PGD attack code
def pgd_attack(original_image, image, alpha, epsilon, data_grad):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + alpha*sign_data_grad
    eta = torch.clamp(perturbed_image - original_image, min=-epsilon, max=epsilon)
    image = torch.clamp(original_image + eta, min=0, max=1).detach_()
    return image