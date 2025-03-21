python -m train.train_mdm_motion_control \
    --save_dir save/humanml_root_key_y \
    --dataset humanml_266 \
    --inpainting_mask root_key_y \
    --lambda_keysmooth 0.0 \
    --num_steps 3000000 \
    --resume_checkpoint ./save/humanml_root_key_y/model.pt \
    # --replace_keypose True \
    # --only_text_condition \
