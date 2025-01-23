"""
This code is used for NeurIPS 2022 paper "Blackbox Attacks via Surrogate Ensemble Search"


https://images.google.com/
https://cloud.google.com/vision/docs/detect-labels-image-client-libraries#client-libraries-usage-python

"""

import io
import os
from pathlib import Path

# Imports the Google Cloud client library
from google.cloud import vision
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions.normal import Normal
from tqdm import tqdm

from class_names_imagenet import lab_dict as imagenet_names
from utils import load_imagenet_1000, load_model, get_adv_np, normalize, get_loss_fn, softmax


def set_log_file(fname):
    import subprocess, sys
    # set log file
    # sys.stdout = os.fdopen(sys.stdout.fileno(), 'wb', 0)
    tee = subprocess.Popen(['tee', fname], stdin=subprocess.PIPE)
    os.dup2(tee.stdin.fileno(), sys.stdout.fileno())
    os.dup2(tee.stdin.fileno(), sys.stderr.fileno())

@torch.no_grad()
def w_regular(w_np_temp_anchor, grad_list, lr_w):
    '''
    w_grad = w_grad - mean(w_grad)/norm(w_grad, 2)
    w = w + lr_w*w_grad
    '''
    w_np_temp_anchor = torch.from_numpy(w_np_temp_anchor)
    w_grad = torch.mean(torch.stack(grad_list), dim=0)
    w_grad_mean = torch.mean(w_grad)
    w_grad = torch.div(torch.sub(w_grad, w_grad_mean), torch.norm(w_grad, p=2))
    w_np_temp_anchor = torch.add(w_np_temp_anchor, torch.mul(w_grad, lr_w))
    w_np_temp_anchor = torch.softmax(w_np_temp_anchor, dim=0)
    return w_np_temp_anchor.numpy()

def get_w_grad(adv, target_idx, w, pert_machine, fuse='loss', untargeted=False, loss_name='ce', maximize=True):
    device = next(pert_machine[0].parameters()).device
        
    target = torch.LongTensor([target_idx]).to(device)
    adv = torch.from_numpy(adv).permute(2,0,1).unsqueeze(0).float().to(device)
    if not isinstance(w, torch.Tensor):
        w = torch.from_numpy(w).float().to(device)
    n_wb = len(pert_machine)
    
    loss_fn = get_loss_fn(loss_name, targeted = not untargeted)
    w.requires_grad = True
    input_tensor = normalize(adv/255)
    outputs = [model(input_tensor) for model in pert_machine]

    if fuse == 'loss':
        loss = sum([w[idx] * loss_fn(outputs[idx],target) for idx in range(n_wb)])
    elif fuse == 'prob':
        target_onehot = F.one_hot(target, 1000)
        prob_weighted = torch.sum(torch.cat([w[idx] * softmax(outputs[idx]) for idx in range(n_wb)], 0), dim=0, keepdim=True)
        loss = - torch.log(torch.sum(target_onehot*prob_weighted))
    elif fuse == 'logit':
        logits_weighted = sum([w[idx] * outputs[idx] for idx in range(n_wb)])
        loss = loss_fn(logits_weighted,target)
    
    if maximize:
        loss = torch.mul(-1, loss)
    loss.backward()
    grad = w.grad
    return grad

# Instantiates a client
client = vision.ImageAnnotatorClient()
def get_gcv_response(file_name, folder):
    """
    Args:
        folder (Path): folder path
        file_name (str): file name

    Returns:
        labels (): response from GCV
    """
    # Loads the image into memory
    with io.open(file_name, 'rb') as image_file:
        content = image_file.read()

    image = vision.Image(content=content)
    # Performs label detection on the image file
    response = client.label_detection(image=image)
    labels = response.label_annotations

    # print gcv response
    lines = []
    for label in labels:
        line = f"{label.description, label.score, label.mid}"
        lines.append(f"{line}\n")
        print(line)
    # write response to txt
    txt_name = f'gcv_{Path(file_name).stem}.txt'
    with open(folder / txt_name, 'w') as f:
        f.writelines(lines)
    return labels


def get_label_set(labels):
    """
    Args: 
        labels (): returned by gcv
    """
    label_set = dict()
    for label in labels:
        label_set[label.description] = label.score
    return label_set


def get_gcv_loss(label_set, label_set_clean):
    """
    Args:
        label_set_clean (dict): the label set of clean images
    """
    success = False
    scores_correct = []
    scores_wrong = []
    for label in label_set:
        if label in label_set_clean:
            scores_correct.append(label_set[label])
        else:
            scores_wrong.append(label_set[label])
    print(f"n_correct: {len(scores_correct)}, sum: {sum(scores_correct)}")
    print(f"n_wrong: {len(scores_wrong)}, sum: {sum(scores_wrong)}")
    # define loss function for gcv
    loss = sum(scores_correct) - sum(scores_wrong)

    # determing success
    def success1():
        # if the top one changes (weakest success)
        # can be more challenging if top 3 switch out (new top 3 and old top 3 do not overlap)
        return list(label_set)[0] != list(label_set_clean)[0]
    def success2():
        # if top 1 is not in the original set (strong success)
        # can be more challenging if top 3 are new
        return list(label_set)[0] not in label_set_clean

    success = success2()
    return loss, success


device = "cuda:3"
img_paths, gt_labels, tgt_labels = load_imagenet_1000(im_root='bilinear_imagenet1000')
def get_im_idx_from_id(im_id):
    """image id
    """
    for im_idx, path in enumerate(img_paths):
        if im_id in path.stem:
            return im_idx


# load surrogate models
n_wb = 20
surrogate_names = ['vgg16_bn', 'resnet18', 'squeezenet1_1', 'googlenet', \
            'mnasnet1_0', 'densenet161', 'efficientnet_b0', \
            'regnet_y_400mf', 'resnext101_32x8d', 'convnext_small', \
            'vgg13', 'resnet50', 'densenet201', 'inception_v3', 'shufflenet_v2_x1_0', \
            'mobilenet_v3_small', 'wide_resnet50_2', 'efficientnet_b4', 'regnet_x_400mf', 'vit_b_16']

wb = []
for model_name in surrogate_names[:n_wb]:
    print(f"load: {model_name}")
    wb.append(load_model(model_name, device))


# attacking hyper-parameters
bound = 'linf'
eps = 12
n_iters = 10
x_alpha = 5
alpha = eps / n_iters
alpha = alpha * x_alpha
untargeted = False
loss_name = 'cw'
n_sampling = 10
noise_std = 0.01

with open('gcv_images/selected_images.txt', 'r') as f:
    data = f.readlines()
im_ids = [i.strip() for i in data]

attack_root = Path('gcv_attack')
success_id_and_count = dict()
success_info_path = attack_root / f"gcv_attack_info_ours.txt"
gaussian_pdf = Normal(0, noise_std)
for idx in tqdm(range(100)):
    im_id = im_ids[idx]
    exp = f'gcv_{im_id}'
    folder = attack_root / exp # folder to store adv images and gcv outputs
    folder.mkdir(parents=True, exist_ok=True)

    im_idx = get_im_idx_from_id(im_id)
    info_gt = f'gt {im_idx}: {gt_labels[im_idx]}, {imagenet_names[gt_labels[im_idx]]}'
    info_tgt = f'tgt {im_idx}: {tgt_labels[im_idx]}, {imagenet_names[tgt_labels[im_idx]]}'
    print(info_gt)
    print(info_tgt)
    with open(folder/f'attack_info_{im_id}.txt', 'w') as f:
        f.writelines([info_gt, '\n', info_tgt])

    clean_path = img_paths[im_idx]
    # query gcv clean image
    print(clean_path)
    labels = get_gcv_response(file_name = clean_path, folder=folder)
    label_set_clean = get_label_set(labels)


    im_np = np.array(Image.open(img_paths[im_idx]).convert('RGB'))
    gt_label = gt_labels[im_idx]
    gt_label_name = imagenet_names[gt_label].split(',')[0]
    tgt_label = tgt_labels[im_idx]
    exp_name = f"idx{im_idx}_f{gt_label}_t{tgt_label}"
    if untargeted:
        tgt_label = gt_label
        exp_name = f"idx{im_idx}_f{gt_label}_untargeted"

    w_np = np.array([1 for _ in range(n_wb)]) / n_wb
    adv_np, losses = get_adv_np(im_np, tgt_label, w_np, wb, bound, eps, n_iters, alpha, untargeted=untargeted, loss_name=loss_name, adv_init=None)
    
    # save png image
    n_query = 1
    adv_name = f"{im_id}_iter{n_query:02d}.png"
    adv_png = Image.fromarray(adv_np.astype(np.uint8))
    adv_png.save(folder/ adv_name)

    # query gcv, first time attack
    print(adv_name)
    labels = get_gcv_response(file_name = folder/adv_name, folder=folder)
    n_query = 1
    label_set_init = get_label_set(labels)
    # initial attack label set
    loss, success = get_gcv_loss(label_set_init, label_set_clean)
    print(f"loss_iter{n_query:02d}: {loss}, success: {success}")

    if success:
        success_id_and_count[im_id] = n_query
        with open(success_info_path, 'a') as f:
            f.write(f"idx: {idx}, {im_id}, counts: {n_query}\n")
        continue

    lr_w = 5e-3
    idx_w = 0
    last_idx = 0
    iterw = 25 # 50
    l2_bound = 0
    while n_query < iterw:
        w_np_temp_anchor = w_np.copy()
        # w update
        grad_list = []
        for _ in range(n_sampling):
            noise = gaussian_pdf.sample((224, 224, 3))
            noise_adv = adv_np + noise.numpy()
            w_grad = get_w_grad(noise_adv, tgt_label, w_np_temp_anchor, wb, untargeted=untargeted, loss_name=loss_name)
            grad_list.append(w_grad.cpu().detach())
        w_np_temp_anchor = w_regular(w_np_temp_anchor, grad_list, lr_w)
        
        adv_np_plus, losses_plus = get_adv_np(im_np, tgt_label, w_np_temp_anchor, wb, bound, eps, n_iters, alpha, untargeted=untargeted, loss_name=loss_name, adv_init=adv_np)
        
        # save png image
        adv_name = f"{im_id}_iter{n_query:02d}.png"
        adv_png = Image.fromarray(adv_np_plus.astype(np.uint8))
        adv_png.save(folder / adv_name)

        # query gcv
        print(adv_name)
        labels = get_gcv_response(file_name = folder/adv_name, folder=folder)
        n_query += 1
        label_set_plus = get_label_set(labels)
        loss_plus, success_plus = get_gcv_loss(label_set_plus, label_set_clean)
        print(f"loss_iter{n_query:02d}: {loss_plus}, success: {success_plus}")
        if success_plus:
            success_id_and_count[im_id] = n_query
            with open(success_info_path, 'a') as f:
                f.write(f"idx: {idx}, {im_id}, counts: {n_query}\n")
            break

        # update
        loss = loss_plus
        w_np = w_np_temp_anchor
        w_np = w_np / w_np.sum()
        last_idx = idx_w
        adv_np = adv_np_plus

        idx_w = (idx_w+1)%n_wb
        if n_query > 5 and last_idx == idx_w:
            lr_w /= 2 # half the lr if there is no change
            print(f"lr_w: {lr_w}")
