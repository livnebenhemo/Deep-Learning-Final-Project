import torch
import torch.nn as nn
import torch.optim as optim
from copy import deepcopy
# https://github.com/Harry24k/CW-pytorch/blob/master/CW.ipynb

# CW-L2 Attack
# Based on the paper, i.e. not exact same version of the code on https://github.com/carlini/nn_robust_attacks
# (1) Binary search method for c, (2) Optimization on tanh space, (3) Choosing method best l2 adversaries is NOT IN THIS CODE.
def cw_l2_attack_one_image(model, mutable_image, constant_image, q_truth, t_truth,
                           device, c=1e-4, max_iter=4, learning_rate=0.01):

    # start image
    w = deepcopy(mutable_image)
    optimizer = optim.Adam([w], lr=learning_rate)
    for step in range(max_iter):
        # calculate cost function

        loss1 = nn.MSELoss(reduction='sum')(w, mutable_image).to(device)
        loss2 = torch.sum(c * f(model, w, constant_image, q_truth, t_truth)).to(device)
        cost = loss1 + loss2
        # minimize the cost function
        optimizer.zero_grad()
        cost.backward()
        optimizer.step()

    return w



def cw_l2_attack_two_images(model, image1, image2, q_truth, t_truth,
                           device, c=1e-4, max_iter=4, learning_rate=0.01):

    # start image
    w1 = deepcopy(image1)
    w2 = deepcopy(image2)
    #TODO:
    # need to check this!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    w = torch.cat((w1, w2), 1)
    #TODO:
    # need to check this!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    image = torch.cat((image1, image2), 1)
    optimizer = optim.Adam([w], lr=learning_rate)
    for step in range(max_iter):
        # calculate cost function

        loss1 = nn.MSELoss(reduction='sum')(w, image).to(device)
        #TODO:
        # need to check this!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        loss2 = torch.sum(c * f(model, w[0: 0.5], w[0.5:1], q_truth, t_truth))
        cost = loss1 + loss2
        # minimize the cost function
        optimizer.zero_grad()
        cost.backward()
        optimizer.step()

    return w[0:0.5], w[0.5:, 1]


# Define f-function
def f(model, image1, image2, q_truth, t_truth):
    q_est, t_est = model(image1, image2)
    return 1/(torch.norm(q_est - q_truth) + torch.norm(t_est - t_truth))

