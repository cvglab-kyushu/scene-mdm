import os, sys
import shutil
import trimesh

sys.path.append(os.getcwd())
from visualize.visualization import render_ply_seq_in_scene, frame2video

if __name__ == "__main__":
    scene_dir = "POSA/POSA_dir/scenes"

    scene_name = "MPH112"
    ply_folder = "./save/humanml_only_text_condition/result_a_person_is_wiping_the_wall_with_a_rag/sample21_rep00_iter=20_converted"
    auto_camera = False
    fps = 20

    savename = '{}_in_{}.mp4'.format(os.path.basename(ply_folder)[:14], scene_name)
    save_path = os.path.join(os.path.dirname(ply_folder), savename)
    save_folder = os.path.join(ply_folder, "imgs")
    scene_path = os.path.join(scene_dir, "{}.ply".format(scene_name))
    scene_data = trimesh.load(scene_path)

    os.makedirs(save_folder, exist_ok=True)
    render_ply_seq_in_scene(save_folder, ply_folder, scene_data, auto_camera=auto_camera)

    frame2video(
        path=os.path.join(save_folder, '%03d.png'),
        video=save_path,
        start=0,
        framerate=fps
    )

    shutil.rmtree(save_folder)
