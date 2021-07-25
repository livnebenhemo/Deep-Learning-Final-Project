import hydra
from experiments.seven_scenes.pipeline import SevenScenesBenchmark
from evaluate.FGSM import evaluate_fgsm_one_image, evaluate_fgsm_two_images
from evaluate.I_FGSM import evaluate_iterative_fgsm_one_image, evaluate_iterative_fgsm_two_images
from evaluate.PGD import evaluate_pgd_one_image, evaluate_pgd_two_images
from evaluate.CW import evaluate_cw_one_image
from evaluate.toggle_I_FGSM import evaluate_toggle_iterative_fgsm_two_images


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

    #evaluate_iterative_fgsm_one_image(benchmark, 0.2, is_distance=True)
    #evaluate_iterative_fgsm_two_images(benchmark, 0.05, is_distance=True)
    #evaluate_pgd_one_image(benchmark, 0.15, is_distance=True)
    #evaluate_cw_one_image(benchmark, 100000, 0.01, is_distance=True)
    #evaluate_toggle_iterative_fgsm_two_images(benchmark, 0.2, is_distance=True)

if __name__ == "__main__":
    main()
