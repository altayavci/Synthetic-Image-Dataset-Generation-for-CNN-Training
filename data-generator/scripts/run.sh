#BASIC 

python3 preprocess.py \
  --dataset data \
  --num_per_aug_img 2 

python3 generate.py \
  --positive_prompt "at the Madison Square Garden"  \
  --num_per_img_prompt 2

python3 outpaint.py \
  --positive_prompt "at the Madison Square Garden"  \
  --scale_factor 2