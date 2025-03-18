import cv2
import numpy as np
import os
import glob


attn_dir = './save/humanml_traj_key_266_y_mask_lkp=1.0_i=7/attn_maps'
layer = 7
# fps = 150
fps = 30
frame_size = (1000, 1000)




image_files = [f"{i:03d}_timestep.png" for i in range(999, -1, -1)]

for folder_path in glob.glob(attn_dir + '/**/'):

    save_path = os.path.join(folder_path, "layer{}_attn.mp4".format(layer))
    out = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, frame_size)

    for idx, image_file in enumerate(image_files):
        image_path = os.path.join(folder_path, f"layer{layer}", image_file)
        frame = cv2.imread(image_path)
        if frame is None:
            continue
        text = f"{999 - idx:03d}"

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 2
        font_color = (0, 0, 0)
        thickness = 2
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        text_x = (frame_size[0] - text_size[0]) // 2
        text_y = text_size[1] + 20  # 少し上に余白を持たせる

        frame = cv2.putText(frame, text, (text_x, text_y), font, font_scale, font_color, thickness, cv2.LINE_AA)
        out.write(frame)
        out.write(frame)

    out.release()