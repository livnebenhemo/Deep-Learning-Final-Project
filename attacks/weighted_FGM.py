import torch


# FGM attack code
def weighted_fgm_attack(image, epsilon, data_grad):
    # compute the absolute values of the gradients
    abs_data_grad = torch.abs(data_grad)
    # compute the sum of all gradients values
    sum_gradients = torch.sum(abs_data_grad)
    # compute the part of each gradient of the sum
    weighted_gradients = torch.div(data_grad, sum_gradients)
    # compute the amount of gradients - we want that the sum of (absolute) changes will be like FGSM = amount*eps
    amount = torch.numel(weighted_gradients)
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + amount*epsilon*weighted_gradients
    # Adding clipping to maintain [0,1] range
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    # Return the perturbed image
    return perturbed_image