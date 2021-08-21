import hydra
from RelPoseNet.experiments.seven_scenes.pipeline import SevenScenesBenchmark
import matplotlib.pyplot as plt
import numpy as np


@hydra.main(config_path="configs", config_name="main")
def main(cfg):
    benchmark = None
    if cfg.experiment.experiment_params.name == '7scenes':
        benchmark = SevenScenesBenchmark(cfg)
        valid_attacks = ['fgsm', 'iterative fgsm', 'c&w', 'pgd', 'toggle iterative fgsm']
        valid_amount_images = ['1', '2', 'one', 'two']

        while True:
            attack = input("Enter the attack you want to run: ")
            if not (attack in valid_attacks):
                print(f'Not a valid attack, try again from the list: {valid_attacks}')
                continue
            break
        while True:
            one_image_or_two_image = input(f'Enter on how many images {attack} attack will do(choose 1 or 2): ')
            if not (one_image_or_two_image in valid_amount_images):
                print(f'Not a valid amount of images, try again from {valid_amount_images}')
                continue
            break

        if attack == "c&w":
            c = float(input("Enter the c you want to run the attack: "))
            lr = float(input("Enter the learning rate you want to run the attack: "))
        else:
            epsilon = float(input("Enter the epsilon you want to run the attack: "))

        if one_image_or_two_image == "1" or one_image_or_two_image == "one":
            if attack == "fgsm":
                print(f'Running attack on fgsm one image with epsilon = {epsilon}')
                evaluate_fgsm_one_image(benchmark, epsilon, is_distance=True)
            elif attack == "iterative fgsm":
                print(f'Running attack on iterative fgsm one image with epsilon = {epsilon}')
                evaluate_iterative_fgsm_one_image(benchmark, epsilon, is_distance=True)
            elif attack == "pgd":
                print(f'Running attack on pgd one image with epsilon = {epsilon}')
                evaluate_pgd_one_image(benchmark, epsilon, is_distance=True)
            elif attack == "c&w":
                print(f'Running attack on c&w one image with c = {c}')
                evaluate_cw_one_image(benchmark, c, learning_rate=lr, is_distance=True)
            else:
                print("The attack is not available, please try another one")

        else:
            if attack == "fgsm":
                print(f'Running attack on fgsm two images with epsilon = {epsilon}')
                evaluate_fgsm_two_images(benchmark, epsilon, is_distance=True)
            elif attack == "iterative fgsm":
                print(f'Running attack on iterative fgsm two images with epsilon = {epsilon}')
                evaluate_iterative_fgsm_two_images(benchmark, epsilon, is_distance=True)
            elif attack == "pgd":
                print(f'Running attack on pgd two images with epsilon = {epsilon}')
                evaluate_pgd_two_images(benchmark, epsilon, is_distance=True)
            elif attack == "toggle fgsm":
                print(f'Running attack on toggle iterative fgsm two images with epsilon = {epsilon}')
                evaluate_toggle_iterative_fgsm_two_images(benchmark, epsilon, is_distance=True)
            else:
                print("The attack is not available, please try another one")

def run_some_epsilons(benchmark):
    examples = []
    epsilons = [0, 0.05, 0.1, 0.15, 0.2]

    # Run test for each epsilon
    for eps in epsilons:
        examples.append(benchmark.evaluate_weighted_fgm_one_image(eps, is_show=True))
    plot_images(epsilons, examples)


# Plot several examples of adversarial samples at each epsilon
def plot_images(epsilons, examples):
    cnt = 0
    plt.figure(figsize=(8, 10))
    for i in range(len(epsilons)):
        for j in range(len(examples[i])):
            cnt += 1
            plt.subplot(len(epsilons), len(examples[0]), cnt)
            plt.xticks([], [])
            plt.yticks([], [])
            if j == 0:
                plt.ylabel("Eps: {}".format(epsilons[i]), fontsize=14)
            ex = examples[i][j]
            plt.imshow(ex)
    plt.tight_layout()
    plt.show()


def plot_graph_by_epsilon():
    # epsilons
    x = [0, 0.05, 0.1, 0.15, 0.2]
    # errors
    y1 = [10.79, 15.04, 17.23, 18.56, 19.48]
    y2 = [0.23, 0.36, 0.39, 0.42, 0.42]

    plt.plot(x, y1, marker='o')
    plt.title("Orientation Error As Function of Epsilon")
    plt.xlabel("epsilon")
    plt.ylabel("orientation error (degree)")
    plt.show()

    plt.plot(x, y2, marker='o')
    plt.title("Translation Error As Function of Epsilon")
    plt.xlabel("epsilon")
    plt.ylabel("translation error (meter)")
    plt.show()


def plot_bar_graph():
    # epsilons
    labels = [0, 0.05, 0.1, 0.15, 0.2]
    x = np.arange(len(labels))
    # one-image fgsm errors
    y1 = [10.79, 11.86, 12.38, 13.09, 13.83]
    y2 = [0.23, 0.31, 0.34, 0.37, 0.38]
    # two-image fgsm errors
    z1 = [10.79, 13.23, 15.21, 17.4, 19.62]
    z2 = [0.23, 0.34, 0.44, 0.45, 0.46]

    # orientation graph
    fig, ax = plt.subplots()
    ax.bar(x, y1, color='b', width=0.25, label="one-image PGD")
    ax.bar(x + 0.25, z1, color='g', width=0.25, label="one-image I-FGSM")

    ax.set_xlabel("epsilon")
    ax.set_ylabel("orientation error (degree)")
    ax.set_title("Orientation Error As Function of Epsilon")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    fig.tight_layout()

    plt.show()

    # translation graph
    fig, ax = plt.subplots()
    ax.bar(x, y2, color='b', width=0.25, label="one-image PGD")
    ax.bar(x + 0.25, z2, color='g', width=0.25, label="one-image I-FGSM")

    ax.set_xlabel("epsilon")
    ax.set_ylabel("translation error (meter)")
    ax.set_title("Translation Error As Function of Epsilon")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    fig.tight_layout()

    plt.show()


def print_distances(benchmark):
    print("------------------------------------- one image -------------------------------")
    for eps in [0.05, 0.1, 0.15, 0.2]:
        print("eps : " + str(eps))
        print(benchmark.evaluate_weighted_fgm_one_image(eps, is_defense=False, is_distance=True))
        print()
    print("------------------------------------- two images -------------------------------")
    for eps in [0.05, 0.1, 0.15, 0.2]:
        print("eps : " + str(eps))
        print(benchmark.evaluate_weighted_fgm_two_images(eps, is_defense=False, is_distance=True))
        print()


if __name__ == "__main__":
    main()
    # plot_bar_graph()
