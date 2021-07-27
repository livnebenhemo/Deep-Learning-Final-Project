from torch import nn
from tqdm import tqdm
import torch
import os
from os import path as osp


# https://pytorch.org/tutorials/beginner/fgsm_tutorial.html
from RelPoseNet.attacks.FGSM import fgsm_attack


def evaluate_fgsm_two_images(self, epsilon, is_defense=False, is_distance=False):
    q_est_all, t_est_all = [], []
    print(f'Evaluate on the dataset...')

    i = 0
    distance = 0
    for data_batch in tqdm(self.dataloader):
        #torch.cuda.empty_cache()
        # Send the data to the device
        data1 = data_batch['img1'].to(self.device)
        data2 = data_batch['img2'].to(self.device)

        # Set requires_grad attribute of tensor. Important for Attack
        for parameter in self.model.parameters():
            parameter.requires_grad = False
        data1.requires_grad = True
        data2.requires_grad = True

        # Forward pass the data through the model
        q_est, t_est = self.model(data1, data2)

        # If the initial prediction is wrong, dont bother attacking, just move on
        """if init_pred.item() != target.item():
            continue""" # to do it?

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
            data2_grad = data2.grad.data

            # Call FGSM Attack - pertubate 1 or 2 images ????
            perturbed_data1 = fgsm_attack(data1, epsilon, data1_grad)
            perturbed_data2 = fgsm_attack(data2, epsilon, data2_grad)

            # Re-estimate the perturbed images
            q_est, t_est = self.model(perturbed_data1, perturbed_data2)
            if is_defense:
                q_est, t_est = self.model(self.eval_augmentation(perturbed_data1), self.eval_augmentation(perturbed_data2))

            if is_distance:
                distance += nn.MSELoss(reduction='mean')(perturbed_data1, data1) + nn.MSELoss(reduction='mean')(perturbed_data2, data2)


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
        with open("fgsm_" + str(epsilon) + "_" + experiment_cfg.output.res_txt_fname, 'w') as f:
            for q_est, t_est in zip(q_est_all, t_est_all):
                f.write(f"{q_est[0]} {q_est[1]} {q_est[2]} {q_est[3]} {t_est[0]} {t_est[1]} {t_est[2]}\n")
    except:
        with open(experiment_cfg.output.res_txt_fname, 'w') as f:
            for q_est, t_est in zip(q_est_all, t_est_all):
                f.write(f"{q_est[0]} {q_est[1]} {q_est[2]} {q_est[3]} {t_est[0]} {t_est[1]} {t_est[2]}\n")

    print(f'Done')

    if is_distance:
        return distance / i


def evaluate_fgsm_one_image(self, epsilon, is_show=False, is_defense=False, is_distance=False):
    q_est_all, t_est_all = [], []
    adv_examples = []
    print(f'Evaluate on the dataset...')

    i = 0
    distance = 0
    for data_batch in tqdm(self.dataloader):
        self.model.eval()
        #torch.cuda.empty_cache()
        # Send the data to the device
        data1 = data_batch['img1'].to(self.device)
        data2 = data_batch['img2'].to(self.device)

        # Set requires_grad attribute of tensor. Important for Attack
        for parameter in self.model.parameters():
            parameter.requires_grad = False
        data1.requires_grad = True

        # Forward pass the data through the model
        q_est, t_est = self.model(data1, data2)

        # If the initial prediction is wrong, dont bother attacking, just move on
        """if init_pred.item() != target.item():
            continue""" # to do it?

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

            # Call FGSM Attack - pertubate 1 or 2 images ????
            perturbed_data1 = fgsm_attack(data1, epsilon, data1_grad)

            if is_show and i%100 == 0:
                # Save some adv examples for visualization later
                if len(adv_examples) < 5:
                    adv_ex = tensor_to_np(perturbed_data1)
                    adv_examples.append(adv_ex)
                if len(adv_examples) == 5:
                    return adv_examples

            # Re-estimate the perturbed images
            """if is_defense:
                self.model.train()"""
            q_est, t_est = self.model(perturbed_data1, data2)
            if is_defense:
                q_est, t_est = self.model(self.eval_augmentation(perturbed_data1), data2)

            if is_distance:
                distance += nn.MSELoss(reduction='mean')(perturbed_data1, data1)

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
