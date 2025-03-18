#!/bin/bash

python finetuned_motion_control.py \
    --model_path "save/humanml_only_text_condition/model.pt" \
    --dataset "humanml" \
    --inpainting_mask "only_text" \
    --guidance_param 2.5 \
    --num_samples 3 \
    --show_input \
    --only_text_condition \
    --motion_length 120 \
    --text_condition "the person walks and sits on a chair."