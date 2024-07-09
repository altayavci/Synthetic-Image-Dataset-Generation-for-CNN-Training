#MULTIVIEW
/data-generator/multiview/bin/python3 multiview.py \
  --dataset data 

/data-generator/general/bin/python3 preprocess.py  \
  --multiview  \
  --dataset scaler/scaled \
  --num_per_aug_img 2     

/data-generator/general/bin/python3 generate.py \
  --positive_prompt "pants, white background, commercial"  \
  --negative_prompt "bad anatomy, ugly, disproportionate body, blur, distorted, split screen" \
  --num_per_img_prompt 2 \
  --captioner_prompt_weight 0.5 \
  --positive_prompt_weight 0.8 

/data-generator/general/bin/python3 outpaint.py \
  --positive_prompt "white background"  \
  --negative_prompt "man, woman, split screen, disproportionate body, blur, distorted" \
  --positive_prompt_weight 0.5  \
  --positive_prompt_suffix_weight 0.8 \
  --negative_prompt_suffix_weight 0.9 \
  --negative_prompt_weight 1.0 \
  --scale_factor 2 \
  --sd_inpainting "Lykon/dreamshaper-8-inpainting"


