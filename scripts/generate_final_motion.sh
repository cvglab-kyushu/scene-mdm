#!/bin/bash

python finetuned_motion_control_posa_trajectory.py \
    --model_path "save/humanml_traj_key_266_y/model.pt" \
    --dataset "humanml_266" \
    --guidance_param 1.0 \
    --num_samples 20 \
    --show_input \
    --inpainting_mask "root_key_y" \
    --posa_output_path "./save/humanml_only_text_condition/result_the_person_walks_and_sits_on_a_chair/sample00_rep00_iter=20_affordance/pkl/MPH11/075.npy" \