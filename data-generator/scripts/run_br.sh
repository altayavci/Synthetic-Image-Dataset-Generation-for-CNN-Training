#BACKGROUND REPLACING

python3 preprocess.py \
  --dataset data \
  --background_replacing \
  --num_per_aug_img 2 

python3 generate.py \
  --positive_prompt "at the Madison Square Garden" \
  --background_replacing \
  --num_per_img_prompt 2 \
  --captioner_prompt_weight 0.6 \
  --positive_prompt_weight 0.8 

python3 outpaint.py \
  --positive_prompt "at the Madison Square Garden"  \
  --positive_prompt_weight 0.6  \
  --positive_prompt_suffix_weight 0.8 \
  --negative_prompt_suffix_weight 0.8 \
  --scale_factor 2