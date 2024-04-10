# Attack on mnasnet0_5 denseNet121 vit_l_16 resnext50_32x4d vgg19


CUDA_VISIBLE_DEVICES=1 python sem_attack.py --n_wb 20 --victim vgg19 -score
