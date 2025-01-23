# Attack on mnasnet0_5 densenet121 vit_l_16

attack_model_name='mnasnet0_5'

echo "attack on ${attack_model_name} ensemble_size=20 image_num=1000"

CUDA_VISIBLE_DEVICES=0 python run_attack.py --n_wb 20 --victim $attack_model_name --loss_name cw

echo "attack on ${attack_model_name} ensemble_size=20 image_num=1000"