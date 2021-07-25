from torch import nn
from tqdm import tqdm
import torch
import os
from os import path as osp
import matplotlib.pyplot as plt
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


# https://github.com/Harry24k/CW-pytorch
from attacks.CW import cw_l2_attack_one_image, cw_l2_attack_two_images


def evaluate_cw_one_image(self, c, learning_rate, is_show=False, is_defense=False, is_distance=False):
    q_est_all, t_est_all = [], []
    adv_examples = []
    print(f'Evaluate on the dataset...')

    i = 0
    distance = 0
    for data_batch in tqdm(self.dataloader):
        # Send the data to the device
        data1 = data_batch['img1'].to(self.device)
        data2 = data_batch['img2'].to(self.device)
        q_gt = data_batch['q_gt'].to(self.device)
        t_gt = data_batch['t_gt'].to(self.device)

        # Set requires_grad attribute of tensor. Important for Attack
        for parameter in self.model.parameters():
            parameter.requires_grad = False
        data1.requires_grad = True

        perturbed_data1 = cw_l2_attack_one_image(self.model, data1, data2, q_gt, t_gt, self.device, c=c, learning_rate=learning_rate)

        adv_ex = perturbed_data1.float().cpu().detach().numpy().squeeze(0).transpose((1, 2, 0))
        plt.imshow(adv_ex)
        plt.show()


        with torch.no_grad():
            if is_show and i % 1 == 0:
                # Save some adv examples for visualization later
                if len(adv_examples) < 5:
                    adv_ex = tensor_to_np(perturbed_data1)
                    adv_examples.append(adv_ex)
                if len(adv_examples) == 5:
                    return adv_examples

            # Re-estimate the perturbed images
            q_est, t_est = self.model(perturbed_data1, data2)
            if is_defense:
                q_est, t_est = self.model(self.eval_augmentation(perturbed_data1), data2)

            if is_distance:
                distance += nn.MSELoss(reduction='mean')(perturbed_data1, data1)

            q_est_all.append(q_est)
            t_est_all.append(t_est)
            i += 1

    q_est_all = torch.cat(q_est_all).cpu().numpy()
    t_est_all = torch.cat(t_est_all).cpu().numpy()

    print(f'Write the estimates to a text file')
    experiment_cfg = self.cfg.experiment.experiment_params

    if not osp.exists(experiment_cfg.output.home_dir):
        os.makedirs(experiment_cfg.output.home_dir)

    try:
        with open("one_image_cw_" + str(learning_rate) + "_" + experiment_cfg.output.res_txt_fname, 'w') as f:
            for q_est, t_est in zip(q_est_all, t_est_all):
                f.write(f"{q_est[0]} {q_est[1]} {q_est[2]} {q_est[3]} {t_est[0]} {t_est[1]} {t_est[2]}\n")
    except:
        with open(experiment_cfg.output.res_txt_fname, 'w') as f:
            for q_est, t_est in zip(q_est_all, t_est_all):
                f.write(f"{q_est[0]} {q_est[1]} {q_est[2]} {q_est[3]} {t_est[0]} {t_est[1]} {t_est[2]}\n")

    print(f'Done')

    if is_distance:
        return distance / i


def evaluate_cw_two_images(self, c, learning_rate, is_show=False, is_defense=False):
    q_est_all, t_est_all = [], []
    adv_examples = []
    print(f'Evaluate on the dataset...')

    i = 0
    for data_batch in tqdm(self.dataloader):
        # Send the data to the device
        data1 = data_batch['img1'].to(self.device)
        data2 = data_batch['img2'].to(self.device)
        q_gt = data_batch['q_gt'].to(self.device)
        t_gt = data_batch['t_gt'].to(self.device)

        # Set requires_grad attribute of tensor. Important for Attack
        for parameter in self.model.parameters():
            parameter.requires_grad = False
        data1.requires_grad = True
        data2.requires_grad = True

        perturbed_data1, perturbed_data2 = cw_l2_attack_two_images(self.model, data1, data2, q_gt, t_gt, self.device, c=c, learning_rate=learning_rate)
        """adv_ex = perturbed_data1.float().cpu().detach().numpy().squeeze(0).transpose((1, 2, 0))
        plt.imshow(adv_ex)
        plt.show()"""
        with torch.no_grad():
            if is_show and i % 100 == 0:
                # Save some adv examples for visualization later
                if len(adv_examples) < 5:
                    adv_ex = tensor_to_np(perturbed_data1)
                    adv_examples.append(adv_ex)
                if len(adv_examples) == 5:
                    return adv_examples

            # Re-estimate the perturbed images
            q_est, t_est = self.model(perturbed_data1, perturbed_data2)
            if is_defense:
                q_est, t_est = self.model(self.eval_augmentation(perturbed_data1), self.eval_augmentation(perturbed_data2))

            q_est_all.append(q_est)
            t_est_all.append(t_est)
            i += 1

    q_est_all = torch.cat(q_est_all).cpu().numpy()
    t_est_all = torch.cat(t_est_all).cpu().numpy()

    print(f'Write the estimates to a text file')
    experiment_cfg = self.cfg.experiment.experiment_params

    if not osp.exists(experiment_cfg.output.home_dir):
        os.makedirs(experiment_cfg.output.home_dir)

    try:
        with open("two_images_cw_" + str(learning_rate) + "_" + experiment_cfg.output.res_txt_fname, 'w') as f:
            for q_est, t_est in zip(q_est_all, t_est_all):
                f.write(f"{q_est[0]} {q_est[1]} {q_est[2]} {q_est[3]} {t_est[0]} {t_est[1]} {t_est[2]}\n")
    except:
        with open(experiment_cfg.output.res_txt_fname, 'w') as f:
            for q_est, t_est in zip(q_est_all, t_est_all):
                f.write(f"{q_est[0]} {q_est[1]} {q_est[2]} {q_est[3]} {t_est[0]} {t_est[1]} {t_est[2]}\n")

    print(f'Done')


def tensor_to_np(tensor):
    img = tensor.float().cpu().numpy().squeeze(0).transpose((1, 2, 0))
    return img
