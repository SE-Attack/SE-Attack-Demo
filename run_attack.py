import argparse
from pathlib import Path

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from PIL import Image
from tqdm import tqdm
from torch.distributions.normal import Normal

from class_names_imagenet import lab_dict as  imagenet_names
from utils import load_imagenet_1000, load_model, get_adv_np, get_label_loss, get_loss_fn, normalize
    
# test
def set_log_file(fname):
    import subprocess, sys, os
    # set log file
    # sys.stdout = os.fdopen(sys.stdout.fileno(), 'wb', 0)
    tee = subprocess.Popen(['tee', fname], stdin=subprocess.PIPE)
    os.dup2(tee.stdin.fileno(), sys.stdout.fileno())
    os.dup2(tee.stdin.fileno(), sys.stderr.fileno())

@torch.no_grad()
def self_updating(adv, target_idx, w, lr_w, pert_machine, untargeted=False, loss_name='ce', noise_sampler=None, sampleing_size=9):
    """
    refine the ensemble weights
    """
    device = next(pert_machine[0].parameters()).device
    # pre-define
    if not isinstance(w, torch.Tensor):
        w = torch.from_numpy(w).float().to(device)
    adv = torch.from_numpy(adv).permute(2,0,1).unsqueeze(0).float().to(device)
    if noise_sampler is not None:
        noise = torch.cat([
            torch.zeros(size=(1, 3, 224, 224), dtype=torch.float), 
            noise_sampler.sample((sampleing_size, 3, 224, 224))
        ]).to(device)
        target = torch.LongTensor([target_idx]*(sampleing_size+1)).to(device)
        input_tensor = normalize(adv/255) + noise
    else:
        target = torch.LongTensor([target_idx]).to(device)
        input_tensor = normalize(adv/255)
    n_wb = len(pert_machine)
    loss_fn = get_loss_fn(loss_name, targeted = not untargeted)

    # forward
    outputs = [model(input_tensor) for model in pert_machine]

    # calculate ensemble loss
    loss = torch.stack([loss_fn(outputs[idx], target) for idx in range(n_wb)])
    if noise_sampler is not None:
        loss = torch.mean(loss, dim=1)
    
    # constuct the update amount
    numerator = loss - torch.mean(loss)
    denominator = torch.max(loss) - torch.min(loss) # Max-min scaling
    update_degree = numerator / denominator

    # refine ensemble weights
    w = w + lr_w*update_degree
    w = torch.softmax(w, dim=0)
    return w.detach().cpu().numpy()


def main():
    parser = argparse.ArgumentParser(description="BASES attack")
    parser.add_argument("--victim", nargs="?", default='vgg19', help="victim model")
    parser.add_argument("--n_wb", type=int, default=10, help="number of models in the ensemble: 4,10,20")
    parser.add_argument("--bound", default='linf', choices=['linf','l2'], help="bound in linf or l2 norm ball")
    parser.add_argument("--eps", type=int, default=16, help="perturbation bound: 10 for linf, 3128 for l2")
    parser.add_argument("--iters", type=int, default=10, help="number of inner iterations: 5,6,10,20...")
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID: 0,1")
    parser.add_argument("--sigma", type=float, default=0, help="the standard deviation of gaussian sampler.")
    parser.add_argument("--root", nargs="?", default='result', help="the folder name of result")
    
    parser.add_argument("--loss_name", nargs="?", default='cw', help="the name of the loss")
    parser.add_argument("--x", type=int, default=3, help="times alpha by x")
    parser.add_argument("--lr", type=float, default=5e-3, help="learning rate of w")
    parser.add_argument("--iterw", type=int, default=50, help="iterations of updating w")
    parser.add_argument("--n_im", type=int, default=1000, help="number of images")
    parser.add_argument("--n_sampling", type=int, default=10, help="number of sampling for MLE")
    parser.add_argument("-untargeted", action='store_true', help="run untargeted attack")

    parser.add_argument("--pid", type=int, default=0)
    parser.add_argument("--num_worker", type=int, default=1)
    args = parser.parse_args()

    device = f'cuda:{args.gpu}'

    lr_w = float(args.lr)
    selfup_round = 1
    sigma = args.sigma
    n_sampling = args.n_sampling

    eps = args.eps
    n_wb = args.n_wb
    bound = args.bound
    n_iters = args.iters
    loss_name = args.loss_name
    alpha = args.x * eps / n_iters

    # load images
    img_paths, gt_labels, tgt_labels = load_imagenet_1000(dataset_root = 'imagenet1000')
    
    # load surrogate models
    surrogate_names = ['vgg16_bn', 'resnet18', 'squeezenet1_1', 'mnasnet1_0', \
                'googlenet', 'densenet161', 'efficientnet_b0', \
                'regnet_y_400mf', 'resnext101_32x8d', 'convnext_small', \
                'vgg13', 'resnet50', 'densenet201', 'inception_v3', 'shufflenet_v2_x1_0', \
                'mobilenet_v3_small', 'wide_resnet50_2', 'efficientnet_b4', 'regnet_x_400mf', 'vit_b_16']
    wb = [] # ensemble surrogate model
    for model_name in surrogate_names[:n_wb]:
        print(f"load: {model_name}")
        wb.append(load_model(model_name, device))

    # load victim model
    victim_model = load_model(args.victim, device)

    # define Gaussian sampler
    if sigma == 0:
        gaussian = None
    else:
        gaussian = Normal(0, sigma) #N(0, 0.1)

    # create exp folders
    exp = f'victim_{args.victim}_{n_wb}wb_sigam{sigma}_{n_sampling}_{bound}_{eps}_iters{n_iters}_x{args.x}_loss_{loss_name}_lr{lr_w}_iterw{args.iterw}'
    if args.untargeted:
        exp = 'untargeted_' + exp

    exp_root = Path(f"selfup_attack_bb_w_logs/") / exp
    exp_root.mkdir(parents=True, exist_ok=True)
    print(exp)
    adv_root = Path(f"selfup_attack_w_adv_images/") / exp
    adv_root.mkdir(parents=True, exist_ok=True)

    # exp records
    success_idx_list = set()
    success_idx_list_pretend = set() # untargeted success
    query_list = []
    query_list_pretend = []
    
    # attack
    for im_idx in tqdm(range(args.pid, args.n_im, args.num_worker)):
        lr_w = float(args.lr) # re-initialize
        im_np = np.array(Image.open(img_paths[im_idx]).convert('RGB'))
        gt_label = gt_labels[im_idx]
        gt_label_name = imagenet_names[gt_label].split(',')[0]
        tgt_label = tgt_labels[im_idx]
        exp_name = f"idx{im_idx}_f{gt_label}_t{tgt_label}"
        if args.untargeted:
            tgt_label = gt_label
            exp_name = f"idx{im_idx}_f{gt_label}_untargeted"
        
        # start from equal weights
        w_np = np.array([1 for _ in range(len(wb))]) / len(wb)
        adv_np, losses = get_adv_np(im_np, tgt_label, w_np, wb, bound, eps, n_iters, alpha, untargeted=args.untargeted, loss_name=loss_name, adv_init=None)
        label_idx, loss, _ = get_label_loss(adv_np/255, victim_model, tgt_label, loss_name, targeted = not args.untargeted)
        n_query = 1
        w_list = []
        loss_wb_list = losses   # loss of optimizing wb models
        loss_bb_list = []       # loss of victim model
        print(f"{label_idx, imagenet_names[label_idx]}, loss: {loss}")
        print(f"w: {w_np.tolist()}")

        # pretend
        if not args.untargeted and label_idx != gt_label:
            success_idx_list_pretend.add(im_idx)
            query_list_pretend.append(n_query)

        if (not args.untargeted and label_idx == tgt_label) or (args.untargeted and label_idx != tgt_label):
            # originally successful
            print('success')
            success_idx_list.add(im_idx)
            query_list.append(n_query)
            w_list.append(w_np.tolist())
            loss_bb_list.append(loss)
        else: 
            while n_query < args.iterw:
                w_np_temp_anchor = w_np.copy()

                # self-updating
                for _ in range(selfup_round):
                    # refine the ensemble weights
                    w_np_temp_anchor = self_updating(adv_np, tgt_label, w_np_temp_anchor, lr_w, wb, 
                                                     untargeted=args.untargeted, loss_name=loss_name, noise_sampler=gaussian, sampleing_size=n_sampling-1)
                    # regenerate adversarial examples
                    adv_np_plus, losses_plus = get_adv_np(im_np, tgt_label, w_np_temp_anchor, wb, bound, eps, n_iters, alpha, 
                                                        untargeted=args.untargeted, loss_name=loss_name, adv_init=adv_np)
                print("w is updated.")
                
                # query
                label_plus, loss_plus, _ = get_label_loss(adv_np_plus/255, victim_model, tgt_label, loss_name, targeted = not args.untargeted)
                n_query += 1
                print(f"iter: {n_query}, {label_plus, imagenet_names[label_plus]}, loss: {loss_plus}")
                
                # pretend
                if (not args.untargeted and label_plus != gt_label) and (im_idx not in success_idx_list_pretend):
                    success_idx_list_pretend.add(im_idx)
                    query_list_pretend.append(n_query)

                # stop if successful
                if (not args.untargeted)*(tgt_label == label_plus) or args.untargeted*(tgt_label != label_plus):
                    print('success')
                    success_idx_list.add(im_idx)
                    query_list.append(n_query)
                    loss = loss_plus
                    w_np = w_np_temp_anchor
                    adv_np = adv_np_plus
                    loss_wb_list += losses_plus
                    break

                # update
                loss = loss_plus
                w_np = w_np_temp_anchor
                adv_np = adv_np_plus
                loss_wb_list += losses_plus

                w_list.append(w_np.tolist())
                loss_bb_list.append(loss)

        # show attack result
        print(f"im_idx: {im_idx}")
        if im_idx in success_idx_list:
            # save to txt
            info = f"im_idx: {im_idx}, iters: {query_list[-1]}, loss: {loss:.2f}, w: {w_np.squeeze().tolist()}\n"
            file = open(exp_root / f'{exp}.txt', 'a')
            file.write(f"{info}")
            file.close()
        print(f"targeted. total_success: {len(success_idx_list)}; success_rate: {len(success_idx_list)/(im_idx+1)}, avg queries: {np.mean(query_list)}")

        if im_idx in success_idx_list_pretend:
            # save to txt
            info = f"im_idx: {im_idx}, iters: {query_list_pretend[-1]}, loss: {loss:.2f}, w: {w_np.squeeze().tolist()}\n"
            file = open(exp_root / f'{exp}_pretend.txt', 'a')
            file.write(f"{info}")
            file.close()
        print(f"untargeted. total_success: {len(success_idx_list_pretend)}; success_rate: {len(success_idx_list_pretend)/(im_idx+1)}, avg queries: {np.mean(query_list_pretend)}")

        # save adv image
        adv_path = adv_root / f"{img_paths[im_idx].name}"
        adv_png = Image.fromarray(adv_np.astype(np.uint8))
        adv_png.save(adv_path)

        # plot figs
        fig, ax = plt.subplots(1,5,figsize=(30,5))
        ax[0].plot(loss_wb_list)
        ax[0].set_xlabel('iters')
        ax[0].set_title('loss on surrogate ensemble')
        ax[1].imshow(im_np)
        ax[1].set_title(f"clean image:\n{gt_label_name}")
        victim_label_idx, loss, _ = get_label_loss(adv_np/255, victim_model, tgt_label, loss_name, targeted = not args.untargeted)
        adv_label_name = imagenet_names[victim_label_idx].split(',')[0]
        ax[2].imshow(adv_np/255)
        ax[2].set_title(f"adv image:\n{adv_label_name}")
        ax[3].plot(loss_bb_list)
        ax[3].set_title('loss on victim model')
        ax[3].set_xlabel('iters')
        ax[4].plot(w_list)
        ax[4].legend(surrogate_names[:n_wb], shadow=True, bbox_to_anchor=(1, 1))
        ax[4].set_title('w of surrogate models')
        ax[4].set_xlabel('iters')
        plt.tight_layout()
        if im_idx in success_idx_list:
            plt.savefig(exp_root / f"{exp_name}_success.png")
        else:
            plt.savefig(exp_root / f"{exp_name}.png")
        plt.close()

        print(f"query_list: {query_list}")
        print(f"avg queries: {np.mean(query_list)}")


if __name__ == '__main__':
    main()
