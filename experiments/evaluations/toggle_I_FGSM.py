import copy

from torch import nn
from tqdm import tqdm
import torch
import os
from os import path as osp

from attacks.iterative_fgsm import ifgsm_attack


def evaluate_toggle_iterative_fgsm_two_images(self, epsilon, is_show=False, is_defense=False, is_distance=False):
    # The paper said min(eps + 4, 1.25*eps) is used as iterations
    iterations_number = 4
    alpha = epsilon / iterations_number

    q_est_all, t_est_all = [], []
    adv_examples = []
    print(f'Evaluate on the dataset...')

    i = 0
    distance = 0
    for data_batch in tqdm(self.dataloader):
        if is_show and i % 200:
            i += 1
            continue
        # Send the data to the device
        data1 = data_batch['img1'].to(self.device)
        data2 = data_batch['img2'].to(self.device)
        original_data1 = copy.deepcopy(data1)
        original_data2 = copy.deepcopy(data2)
        for j in range(iterations_number):
            # Set requires_grad attribute of tensor. Important for Attack
            for parameter in self.model.parameters():
                parameter.requires_grad = False
            data1.requires_grad = True
            data2.requires_grad = True

            # Forward pass the data through the model
            q_est, t_est = self.model(data1, data2)

            # Calculate the loss
            loss, t_loss_val, q_loss_val = self.criterion(data_batch['q_gt'].to(self.device),
                                                          data_batch['t_gt'].to(self.device),
                                                          q_est,
                                                          t_est)

            del q_est, t_est, t_loss_val, q_loss_val

            # Zero all existing gradients
            self.model.zero_grad()

            # Calculate gradients of model in backward pass
            loss.backward()

            with torch.no_grad():
                # Collect datagrad
                data1_grad = data1.grad.data
                # Call I-FGSM Attack
                data1 = ifgsm_attack(data1, alpha, data1_grad)

            # Forward pass the data through the model
            q_est, t_est = self.model(data1, data2)

            # Calculate the loss
            loss, t_loss_val, q_loss_val = self.criterion(data_batch['q_gt'].to(self.device),
                                                          data_batch['t_gt'].to(self.device),
                                                          q_est,
                                                          t_est)

            del q_est, t_est, t_loss_val, q_loss_val

            # Zero all existing gradients
            self.model.zero_grad()

            # Calculate gradients of model in backward pass
            loss.backward()

            with torch.no_grad():
                # Collect datagrad
                data2_grad = data2.grad.data
                # Call I-FGSM Attack
                data2 = ifgsm_attack(data2, alpha, data2_grad)

        with torch.no_grad():
            if is_show and i % 100 == 0:
                # Save some adv examples for visualization later
                if len(adv_examples) < 5:
                    adv_ex = tensor_to_np(data1)
                    adv_examples.append(adv_ex)
                if len(adv_examples) == 5:
                    return adv_examples

            # Re-estimate the perturbed images
            q_est, t_est = self.model(data1, data2)
            if is_defense:
                q_est, t_est = self.model(self.eval_augmentation(data1), self.eval_augmentation(data2))

            if is_distance:
                distance += nn.MSELoss(reduction='mean')(original_data1, data1) + nn.MSELoss(reduction='mean')(original_data2, data2)

            q_est_all.append(q_est)
            t_est_all.append(t_est)
            i += 1

            if is_distance and i == 200:
                break

    q_est_all = torch.cat(q_est_all).cpu().numpy()
    t_est_all = torch.cat(t_est_all).cpu().numpy()

    print(f'Write the estimates to a text file')
    experiment_cfg = self.cfg.experiment.experiment_params

    if not osp.exists(experiment_cfg.output.home_dir):
        os.makedirs(experiment_cfg.output.home_dir)

    try:
        with open("one_image_fgsm_" + str(epsilon) + "_" + experiment_cfg.output.res_txt_fname, 'w') as f:
            for q_est, t_est in zip(q_est_all, t_est_all):
                f.write(f"{q_est[0]} {q_est[1]} {q_est[2]} {q_est[3]} {t_est[0]} {t_est[1]} {t_est[2]}\n")
    except:
        with open(experiment_cfg.output.res_txt_fname, 'w') as f:
            for q_est, t_est in zip(q_est_all, t_est_all):
                f.write(f"{q_est[0]} {q_est[1]} {q_est[2]} {q_est[3]} {t_est[0]} {t_est[1]} {t_est[2]}\n")

    print(f'Done')

    if is_distance:
        return distance / i


def tensor_to_np(tensor):
    # img = tensor.mul(255).byte()
    img = tensor.float().cpu().numpy().squeeze(0).transpose((1, 2, 0))
    return img
