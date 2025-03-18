import cv2
import os

# 動画が保存されているフォルダのパスを設定
folder_path = './save/humanml_traj_key_266_y_mask_keysmooth/result_a_person_walks_and_sits_on_a_chair_gp=0.6'

# フォルダ内のすべてのファイルを取得
file_list = os.listdir(folder_path)

# 動画ファイルのみをフィルタリング
video_files = [f for f in file_list if f.endswith('.mp4')]

# 各動画ファイルに対して処理を行う
for video_file in video_files:
    video_path = os.path.join(folder_path, video_file)
    
    # 動画を読み込み
    cap = cv2.VideoCapture(video_path)
    
    # 動画のプロパティを取得
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 保存する動画ファイルの設定
    output_path = os.path.join(folder_path, f'{video_file}')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 動画コーデックの設定
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    
    frame_number = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        text = f'Frame: {frame_number}'
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.4  # フォントサイズ
        font_color = (0, 0, 0)  # フォントカラー（白）
        thickness = 1  # フォントの厚さ
        size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        position = (frame_width - size[0] - 10, size[1] + 10)  # 右上の位置調整
        
        # テキストを描画
        cv2.putText(frame, text, position, font, font_scale, font_color, thickness, cv2.LINE_AA)
        
        # フレームを保存
        out.write(frame)

        frame_number += 1
    
    # 動画を閉じる
    cap.release()
    out.release()

print("Processing complete.")