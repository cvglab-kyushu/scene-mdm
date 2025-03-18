python -m train.train_mdm_motion_control \
    --save_dir save/humanml_key_joint_266_mask_keysmooth=1.0_wo_velocity \
    --dataset humanml_mask \
    --inpainting_mask key_joint \
    --lambda_keysmooth 1.0 \
    --num_steps 3000000 \
    # --resume_checkpoint ./save/humanml_key_joint_266_mask_keysmooth=1.0/model.pt \
    # --replace_keypose True \
    # --only_text_condition \
    # --inpainting_mask root_key_y \
