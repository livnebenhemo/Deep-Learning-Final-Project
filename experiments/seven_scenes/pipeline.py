import os
from os import path as osp
from numba import cuda
from tqdm import tqdm
import torch

from RelPoseNet.experiments.service.benchmark_base import Benchmark
from RelPoseNet.relposenet.criterion import RelPoseCriterion
from RelPoseNet.relposenet.dataset import SevenScenesTestDataset
from RelPoseNet.relposenet.augmentations import get_augmentations, get_defense_augmentation
from RelPoseNet.relposenet.model import RelPoseNet
import RelPoseNet.experiments.evaluations.FGSM
import RelPoseNet.experiments.evaluations.I_FGSM
import RelPoseNet.experiments.evaluations.toggle_I_FGSM
import RelPoseNet.experiments.evaluations.FGM
import RelPoseNet.experiments.evaluations.weighted_FGM
import RelPoseNet.experiments.evaluations.PGD
import RelPoseNet.experiments.evaluations.CW


class SevenScenesBenchmark(Benchmark):
    def __init__(self, cfg):
        super().__init__(cfg)

        self.dataloader = self._init_dataloader()
        self.model = self._load_model_relposenet().to(self.device)
        # Criterion <==> loss
        self.criterion = RelPoseCriterion().to(self.device)
        self.eval_augmentation = get_defense_augmentation()

    def _init_dataloader(self):
        experiment_cfg = self.cfg.experiment.experiment_params

        # define test augmentations
        _, eval_aug = get_augmentations()

        # test dataset
        dataset = SevenScenesTestDataset(experiment_cfg, eval_aug)

        # define a dataloader
        dataloader = torch.utils.data.DataLoader(dataset,
                                                 # batch_size=experiment_cfg.bs,
                                                 batch_size=1,
                                                 shuffle=False,
                                                 pin_memory=True,
                                                 num_workers=getattr(cuda.get_current_device(), 'MULTIPROCESSOR_COUNT'),
                                                 drop_last=False)

        return dataloader

    def _load_model_relposenet(self):
        print(f'Loading RelPoseNet model...')
        model_params_cfg = self.cfg.model.model_params
        #model = RelPoseNet(model_params_cfg)
        model = RelPoseNet.relposenet.model.RelPoseNet(model_params_cfg)

        data_dict = torch.load(model_params_cfg.snapshot)
        model.load_state_dict(data_dict['state_dict'])
        print(f'Loading RelPoseNet model... Done!')
        return model.eval()

    def evaluate(self):
        q_est_all, t_est_all = [], []
        print(f'Evaluate on the dataset...')
        with torch.no_grad():
            for data_batch in tqdm(self.dataloader):
                q_est, t_est = self.model(data_batch['img1'].to(self.device),
                                          data_batch['img2'].to(self.device))

                q_est_all.append(q_est)
                t_est_all.append(t_est)

        q_est_all = torch.cat(q_est_all).cpu().numpy()
        t_est_all = torch.cat(t_est_all).cpu().numpy()

        print(f'Write the estimates to a text file')
        experiment_cfg = self.cfg.experiment.experiment_params

        if not osp.exists(experiment_cfg.output.home_dir):
            os.makedirs(experiment_cfg.output.home_dir)

        with open(experiment_cfg.output.res_txt_fname, 'w') as f:
            for q_est, t_est in zip(q_est_all, t_est_all):
                f.write(f"{q_est[0]} {q_est[1]} {q_est[2]} {q_est[3]} {t_est[0]} {t_est[1]} {t_est[2]}\n")

        print(f'Done')

    # https://pytorch.org/tutorials/beginner/fgsm_tutorial.html
    def evaluate_fgsm_two_images(self, epsilon, is_defense=False, is_distance=False):
        return RelPoseNet.experiments.evaluations.FGSM.evaluate_fgsm_two_images(self, epsilon, is_defense, is_distance)

    def evaluate_fgsm_one_image(self, epsilon, is_show=False, is_defense=False, is_distance=False):
        return RelPoseNet.experiments.evaluations.FGSM.evaluate_fgsm_one_image(self, epsilon, is_show, is_defense, is_distance)

    # https://pytorch.org/tutorials/beginner/fgsm_tutorial.html
    def evaluate_fgm_two_images(self, epsilon, is_defense=False):
        return RelPoseNet.experiments.evaluations.FGM.evaluate_fgm_two_images(self, epsilon, is_defense)

    def evaluate_fgm_one_image(self, epsilon, is_defense=False):
        return RelPoseNet.experiments.evaluations.FGM.evaluate_fgm_one_image(self, epsilon, is_defense)

    def evaluate_weighted_fgm_two_images(self, epsilon, is_defense=False, is_distance=False):
        return RelPoseNet.experiments.evaluations.weighted_FGM.evaluate_weighted_fgm_two_images(self, epsilon, is_defense, is_distance)

    def evaluate_weighted_fgm_one_image(self, epsilon,is_show=False, is_defense=False, is_distance=False):
        return RelPoseNet.experiments.evaluations.weighted_FGM.evaluate_weighted_fgm_one_image(self, epsilon, is_show, is_defense, is_distance)

    # https://arxiv.org/pdf/1607.02533.pdf
    # https://github.com/Harry24k/AEPW-pytorch/blob/master/Adversarial%20examples%20in%20the%20physical%20world.ipynb
    def evaluate_iterative_fgsm_one_image(self, epsilon, is_show=False, is_defense=False, is_distance=False):
        return RelPoseNet.experiments.evaluations.I_FGSM.evaluate_iterative_fgsm_one_image(self, epsilon, is_show, is_defense, is_distance)

    def evaluate_iterative_fgsm_two_images(self, epsilon, is_show=False, is_defense=False, is_distance=False):
        return RelPoseNet.experiments.evaluations.I_FGSM.evaluate_iterative_fgsm_two_images(self, epsilon, is_show, is_defense, is_distance)

    def evaluate_toggle_iterative_fgsm_two_images(self, epsilon, is_show=False, is_defense=False, is_distance=False):
        return RelPoseNet.experiments.evaluations.toggle_I_FGSM.evaluate_toggle_iterative_fgsm_two_images(self, epsilon, is_show, is_defense, is_distance)

    # https://github.com/Harry24k/PGD-pytorch
    def evaluate_pgd_one_image(self, epsilon, is_show=False, is_defense=False, is_distance=False):
        return RelPoseNet.experiments.evaluations.PGD.evaluate_pgd_one_image(self, epsilon, is_show, is_defense, is_distance)

    def evaluate_pgd_two_images(self, epsilon, is_show=False, is_defense=False, is_distance=False):
        return RelPoseNet.experiments.evaluations.PGD.evaluate_pgd_two_images(self, epsilon, is_show, is_defense, is_distance)

    def evaluate_cw_one_image(self, c, learning_rate, is_show=False, is_defense=False, is_distance=False):
        return RelPoseNet.experiments.evaluations.CW.evaluate_cw_one_image(self, c, learning_rate, is_show, is_defense, is_distance)


def tensor_to_np(tensor):
    # img = tensor.mul(255).byte()
    img = tensor.float().cpu().numpy().squeeze(0).transpose((1, 2, 0))
    return img
