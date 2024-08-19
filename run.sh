# Attack on mnasnet0_5 densenet121 vit_l_16 efficientnet_b0 convnext_base vgg19

attack_model_name='mnasnet0_5'

echo "attack on ${attack_model_name} ensemble_size=20 image_num=1000"

CUDA_VISIBLE_DEVICES=1 python my_query_w_bb.py --n_wb 20 --victim $attack_model_name --loss_name cw

echo "attack on ${attack_model_name} ensemble_size=20 image_num=1000"