# Text-Guided Diverse Scene Interaction Synthesis by Disentangling Actions from Scenes

## Getting started

This code was tested on `Windows 11` and requires:

* Python 3.9.19
* PyTorch 2.2.1
* conda3 or miniconda3
* CUDA capable GPU (one is enough)

### 1. Setup environment 

Install ffmpeg (if not already installed):

```shell
sudo apt update
sudo apt install ffmpeg
```
For windows use [this](https://www.geeksforgeeks.org/how-to-install-ffmpeg-on-windows/) instead.

Setup conda env:
```shell
conda create -n scene-mdm python==3.9.19
conda activate scene-mdm
python -m spacy download en_core_web_sm
pip install git+https://github.com/openai/CLIP.git
pip install git+https://github.com/GuyTevet/smplx.git
```

Install PyTorch from [here](https://pytorch.org/get-started/previous-versions/).

```shell
pip install -r requirements.txt
```

### 2. Get MDM dependencies

PriorMDM share most of its dependencies with the original MDM. 
If you already have an installed MDM from the official repo, you can save time and link the dependencies instead of getting them from scratch.

**Download dependencies:**

```bash
bash prepare/download_smpl_files.sh
bash prepare/download_glove.sh
bash prepare/download_t2m_evaluators.sh
```

Download the SMPLX model weights from [here](https://download.is.tue.mpg.de/download.php?domain=smplx&sfile=models_smplx_v1_1.zip) and put to `./body_models/smplx"`.

**Get HumanML3D dataset**:

Follow the instructions in [HumanML3D](https://github.com/EricGuo5513/HumanML3D.git),
then copy the result dataset to our repository:

```shell
cp -r ../HumanML3D/HumanML3D ./dataset/HumanML3D
```

**Get POSA data** 
Download the [POSA_dir.zip](https://posa.is.tue.mpg.de/) and put it to `./POSA/POSA_dir`.

**Get pretrained model**
Download T2M model from [here]() and SceneMDM model from [here]() and put it on `./save/humanml_only_text_condition` and `./save/humanml_traj_key_266_y` respectively.



## Training SceneMDM

```shell
bash run_train.sh
```

## Motion Generation

Please run the following shell scripts in order.

1. **scripts/generate_temporal_motion.sh**  
   Generates a temporary motion from text. Modify `--text_condition` in the script to any desired sentence.

2. **scripts/run_fit_seq.sh**  
   Converts the motion generated in step 1 into an SMPL-X format mesh. Modify `--input_path` in the script to the path of the motion file generated in step 1.
   <details>
      <summary>If you are faced with <code>IndexError: index **** is out of bounds for dimension 0 with size 6890</code></summary>
      Change line 231 and after in envs/scene-mdm/lib/site-packages/smplx/lbs.py as follows:
      
      ```python
         J_transformed, A = batch_rigid_transform(rot_mats, J, parents, dtype=dtype)

         # 5. Do skinning:
         # W is N x V x (J + 1)
         W = lbs_weights.unsqueeze(dim=0).expand([batch_size, -1, -1])
         # (N x V x (J + 1)) x (N x (J + 1) x 16)
         num_joints = J_regressor.shape[0]
         T = torch.matmul(W, A.view(batch_size, num_joints, 16)) \
            .view(batch_size, -1, 4, 4)

         homogen_coord = torch.ones([batch_size, v_posed.shape[1], 1],
                                    dtype=dtype, device=device)
         v_posed_homo = torch.cat([v_posed, homogen_coord], dim=2)
         v_homo = torch.matmul(T, torch.unsqueeze(v_posed_homo, dim=-1))

         verts = v_homo[:, :, :3, 0]

         return verts, J_transformed
      ```
   </details>

3. **scripts/run_OS-POSA.sh**  
   Runs Path Planning and OS-POSA. Modify `--mdm_out_dir` in the script to the path of the folder generated in step 2. Also, change `--scene_name` to any desired PROX scene name.

4. **scripts/generate_final_motion.sh**  
   Runs SceneMDM. Modify `--posa_output_path` to the path of the npy file output in step 3.

5. **scripts/run_fit_seq.sh**  
   Converts the motion generated in step 4 into an SMPL-X format mesh. Modify `--input_path` in the script to the path of the motion file generated in step 4.

6. **scripts/run_convert_vertex.sh**  
   The SMPL mesh output in 5 is rigidly transformed to the coordinate system of the PROX scene. Modify the argument `--mdm_out_dir` to the path to the folder output in 5.


## Visualization Method

Blender is used for visualization. Follow these steps to import human motion animation.

1. Download "stop-motion-obj-v2.1.1.zip" from the following link:  
   [Stop-motion-OBJ v2.1.1](https://github.com/neverhood311/Stop-motion-OBJ/releases/tag/v2.1.1)

2. Open Blender, go to **"Edit" → "Preferences" → "Install"**, and select the downloaded ZIP file.

3. Enable the installed **Stop-motion-OBJ** add-on. In **Preferences**, go to the **Add-ons** tab, type "stop" in the search box, and check the displayed **Stop-motion-OBJ** option.

4. If the add-on is successfully installed and enabled, you should see **"Mesh Sequence"** under **"File" → "Import"**. Click on it.

5. In the **File Settings** panel on the right, set **Transform** to **Y Forward, Z Up**.

6. Open the folder containing the OBJ files and enter the common filename in the **File Name** field at the bottom right.

7. Click the **"Select Folder"** button to load the sequence of OBJ files.

Please import the scene you specified as input.





## Acknowledgments

This code is standing on the shoulders of giants. We want to thank the following contributors
that our code is based on:

[PriorMDM](https://github.com/priorMDM/priorMDM), 
[HumanML3D](https://github.com/EricGuo5513/HumanML3D), 
[POSA](https://github.com/mohamedhassanmus/POSA), 

## Bibtex
If you find this code useful in your research, please cite:
<!-- 
```
@article{shafir2023human,
  title={Human motion diffusion as a generative prior},
  author={Shafir, Yonatan and Tevet, Guy and Kapon, Roy and Bermano, Amit H},
  journal={arXiv preprint arXiv:2303.01418},
  year={2023}
}
```
 -->
