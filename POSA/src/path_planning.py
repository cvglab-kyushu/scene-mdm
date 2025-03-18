import os
import glob
import copy
import json
import pickle
import torch

import numpy as np
import trimesh
import pyrender
from pathlib import Path
from pathfinder import navmesh_baker as nmb
import pathfinder as pf
from scipy.spatial.distance import cdist
from copy import deepcopy
from src.smplx2humanml import convert_smplx2humanml
from human_body_prior.body_model.body_model import BodyModel


zup_to_shapenet = np.array(
    [[1, 0, 0, 0],
     [0, 0, 1, 0],
     [0, -1, 0, 0],
     [0, 0, 0, 1]]
)
shapenet_to_zup = np.array(
    [[1, 0, 0, 0],
     [0, 0, -1, 0],
     [0, 1, 0, 0],
     [0, 0, 0, 1]]
)

unity_to_zup = np.array(
            [[-1, 0, 0, 0],
             [0, 0, -1, 0],
             [0, 1, 0, 0],
             [0, 0, 0, 1]]
        )

def triangulate(vertices, polygons):
    triangle_faces = []
    for face in polygons:
        for idx in range(len(face) - 2):
            triangle_faces.append((face[0], face[idx + 1], face[idx + 2]))
    return trimesh.Trimesh(vertices=np.array(vertices),
                           faces=np.array(triangle_faces),
                           vertex_colors=np.array([0, 0, 200, 100]))

def rotate_vector(vector, axis, angle):
    """
    ベクトルを指定した軸(axis)周りに回転させる。
    
    :param vector: 回転させる元のベクトル（正規化されていること）
    :param axis: 回転軸となるベクトル（正規化されていること）
    :param angle: 回転角度（ラジアン）
    :return: 回転後のベクトル
    """
    axis = np.array(axis)
    vector = np.array(vector)
    
    # 回転行列の計算 (ロドリゲスの回転公式)
    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)
    
    # 回転行列の要素
    rotation_matrix = np.array([
        [cos_angle + axis[0]**2 * (1 - cos_angle), 
         axis[0] * axis[1] * (1 - cos_angle) - axis[2] * sin_angle, 
         axis[0] * axis[2] * (1 - cos_angle) + axis[1] * sin_angle],
        
        [axis[1] * axis[0] * (1 - cos_angle) + axis[2] * sin_angle, 
         cos_angle + axis[1]**2 * (1 - cos_angle), 
         axis[1] * axis[2] * (1 - cos_angle) - axis[0] * sin_angle],
        
        [axis[2] * axis[0] * (1 - cos_angle) - axis[1] * sin_angle, 
         axis[2] * axis[1] * (1 - cos_angle) + axis[0] * sin_angle, 
         cos_angle + axis[2]**2 * (1 - cos_angle)]
    ])
    
    # 回転後のベクトル
    rotated_vector = np.dot(rotation_matrix, vector)
    
    return rotated_vector

def create_navmesh(scene_mesh, export_path, agent_radius=0.2, agent_height=1.0,
                    agent_max_climb=0.1, agent_max_slope=15.0,
                   visualize=False):
    baker = nmb.NavmeshBaker()
    vertices = scene_mesh.vertices.tolist()
    vertices = [tuple(vertex) for vertex in vertices]
    faces = scene_mesh.faces.tolist()
    baker.add_geometry(vertices, faces)
    # bake navigation mesh
    baker.bake(
        verts_per_poly=3,
        cell_size=0.05, cell_height=0.05,
        agent_radius=agent_radius,
        agent_height=agent_height,
        agent_max_climb=agent_max_climb, agent_max_slope=agent_max_slope
    )
    # obtain polygonal description of the mesh
    vertices, polygons = baker.get_polygonization()
    triangulated = triangulate(vertices, polygons)
    triangulated.apply_transform(shapenet_to_zup)
    # triangulated = triangulated.slice_plane(np.array([0, 0, 0.1]), np.array([0, 0, -1.0]))  # cut off floor faces
    triangulated.vertices[:, 2] = 0
    triangulated.vertices[:, 2] += 0.4
    triangulated.visual.vertex_colors = np.array([0, 0, 200, 100])
    os.makedirs(os.path.dirname(export_path), exist_ok=True)
    triangulated.export(export_path)
    if visualize:
        scene = pyrender.Scene()
        scene_mesh = deepcopy(scene_mesh)
        scene_mesh.vertices[:, 1] -= 0.05
        scene_mesh.apply_transform(shapenet_to_zup)
        scene.add(pyrender.Mesh.from_trimesh(scene_mesh))
        scene.add(pyrender.Mesh.from_trimesh(triangulated))
        pyrender.Viewer(scene, use_raymond_lighting=True)
    return triangulated

"""return can be empty when no path"""
def path_find(navmesh_zup, start_zup, finish_zup, visualize=False, scene_mesh=None):
    # to yup
    start = tuple(start_zup.squeeze().tolist())
    finish = tuple(finish_zup.squeeze().tolist())
    start = (start[0], start[2], -start[1])
    finish = (finish[0], finish[2], -finish[1])
    navmesh = deepcopy(navmesh_zup)
    navmesh.apply_transform(zup_to_shapenet)


    vertices = navmesh.vertices.tolist()
    vertices = [tuple(vertex) for vertex in vertices]
    faces = navmesh.faces.tolist()
    pathfinder = pf.PathFinder(vertices, faces)
    path = pathfinder.search_path(start, finish)
    path = np.array(path)
    if len(path) > 0:
        # to zup
        path = np.stack([path[:, 0], -path[:, 2], path[:, 1],], axis=1)
    else:
        path = np.array([start_zup[0], finish_zup[0]])

    if visualize:
        scene = pyrender.Scene()
        if scene_mesh is not None:
            scene.add(pyrender.Mesh.from_trimesh(scene_mesh))
        scene.add(pyrender.Mesh.from_trimesh(navmesh_zup, poses=np.array([[1, 0, 0, 0],
                                                                      [0, 1, 0, 0, ],
                                                                      [0, 0, 1, 0.05],
                                                                      [0, 0, 0, 1]])))
        vis_meshes = []
        point = trimesh.creation.uv_sphere(radius=0.05)
        point.visual.vertex_colors = np.array([255, 0, 0, 255])
        point.vertices += np.array(start_zup)
        vis_meshes.append(point)
        point = trimesh.creation.uv_sphere(radius=0.05)
        point.visual.vertex_colors = np.array([255, 0, 0, 255])
        point.vertices += np.array(finish_zup)
        vis_meshes.append(point)
        for idx, waypoint in enumerate(path):
            point = trimesh.creation.uv_sphere(radius=0.05)
            point.visual.vertex_colors = np.array([255, 0, 0, 255])
            point.vertices += waypoint
            vis_meshes.append(point)
            if idx > 0:
                line = trimesh.creation.cylinder(radius=0.03, segment=path[idx - 1:idx + 1, :], )
                line.visual.vertex_colors = np.array([0, 255, 0, 255])
                vis_meshes.append(line)
        scene.add(pyrender.Mesh.from_trimesh(trimesh.util.concatenate(vis_meshes)))
        pyrender.Viewer(scene, use_raymond_lighting=True)
    return path


def get_connected_faces(mesh, ind):
    """
    指定した面を含む、途切れていない一つの平面のTrimeshを作成

    Args:
        mesh: Trimeshオブジェクト
        ind: 面のインデックス

    Returns:
        新しいTrimeshオブジェクト
    """
    # すでに訪問した面のインデックスを追跡するためのセット
    visited_faces = set()
    # 探索のためのキュー（初期面を追加）
    queue = [ind]

    # BFS（幅優先探索）で連結面を探索
    while queue:
        current_face = queue.pop(0)  # キューの先頭から面を取得
        current_face = int(current_face)
        
        if current_face not in visited_faces:
            # 現在の面を訪問済みとして記録
            visited_faces.add(current_face)

            # 現在の面の頂点を取得
            current_vertices = mesh.faces[current_face]
            
            # 各頂点に対して、接続された面を探索
            for vertex in current_vertices:
                # その頂点を持つ全ての面を取得
                adjacent_faces = mesh.vertex_faces[vertex]
                for adjacent in adjacent_faces:
                    if adjacent >= 0 and int(adjacent) not in visited_faces:
                        queue.append(adjacent)

    # 訪問済みの面だけで新しいメッシュを作成
    connected_faces = list(visited_faces)
    connected_mesh = mesh.submesh([connected_faces], append=True)

    return connected_mesh

def point_on_circle(point, r, vector):
    # 中心点からベクトルの方向に半径r分進んだ点を計算
    new_x = point[0][0] + r * vector[0]
    new_y = point[0][1] + r * vector[1]
    return np.array([[new_x, new_y, point[0][2]]])

def select_circle_point(navmesh, center, r, vector=[1,0,0], n_rot=20):
    # centerを中心に、rの半径の円上の点を選択
    vector = np.array(vector)
    for i in range(n_rot):
        vector = rotate_vector(vector, axis=[0,0,1], angle=i*2*np.pi/n_rot)
        poc = point_on_circle(center, r, vector)
        _, distance, _  =  navmesh.nearest.on_surface(poc)
        if distance < 1e-4:
            return poc
    return -1

def interpolate_path(path, speed):
    # 経由地点の数
    num_waypoints = len(path)
    
    # スピードの長さ
    num_frames = len(speed)
    
    # 補間後の経路地点を格納するリスト
    interpolated_path = []
    
    # 経由地点間のスピード累積を計算
    distances = [np.linalg.norm(path[i] - path[i-1]) for i in range(1, num_waypoints)]
    total_distance = sum(distances)
    
    # スピードの累積距離を計算
    cumulative_speeds = np.cumsum(speed)
    
    # 各フレームの位置を計算
    current_distance = 0
    waypoint_index = 0
    for i in range(num_frames):
        # 目標距離
        target_distance = cumulative_speeds[i] / cumulative_speeds[-1] * total_distance
        
        # 経由地点間で補間
        while waypoint_index < num_waypoints - 1 and current_distance + distances[waypoint_index] < target_distance:
            current_distance += distances[waypoint_index]
            waypoint_index += 1
        
        if waypoint_index == num_waypoints - 1:
            # ゴール地点に達した場合
            interpolated_path.append(path[-1])
        else:
            # 線形補間
            start_point = path[waypoint_index]
            end_point = path[waypoint_index + 1]
            segment_distance = distances[waypoint_index]
            remaining_distance = target_distance - current_distance
            
            interpolation_factor = remaining_distance / segment_distance
            interpolated_point = start_point + interpolation_factor * (end_point - start_point)
            interpolated_path.append(interpolated_point)
    
    return np.array(interpolated_path)

def is_point_in_surface(point, mesh, tolerance=1e-3):
    vertices = mesh.vertices
    faces = mesh.faces
    triangles = [(vertices[faces[i, 0]], vertices[faces[i, 1]], vertices[faces[i, 2]]) for i in range(len(faces))]
    is_on_surface = False
    for triangle in triangles:
        # 三角形の3つの頂点
        p1, p2, p3 = triangle
        
        # 三角形の法線ベクトルを計算
        normal = np.cross(p2 - p1, p3 - p1)
        normal /= np.linalg.norm(normal)
        
        # 点と三角形の平面までの距離を計算
        distance = np.dot(normal, point - p1)
        
        # 点が平面上にあるかをチェック
        if abs(distance) < tolerance:
            # 点が三角形の範囲内にあるかを判定
            u = p2 - p1
            v = p3 - p1
            w = point - p1

            uu = np.dot(u, u)
            uv = np.dot(u, v)
            vv = np.dot(v, v)
            wu = np.dot(w, u)
            wv = np.dot(w, v)
            
            denominator = uv * uv - uu * vv
            
            # 三角形内部のバリデーションを行うための係数
            s = (uv * wv - vv * wu) / denominator
            t = (uv * wu - uu * wv) / denominator

            if (s >= 0) and (t >= 0) and (s + t <= 1):
                is_on_surface = True
                break

    return is_on_surface

def path_planning_prox(key_data_path=None, navmesh_dir=None, scene_dir=None, scale_factor=0.5):
    '''
    scale_factor: Scale factor when the speed of the original motion is reflected in the path (adjustment required)
    '''

    key_data_dir = os.path.dirname(key_data_path)
    key_data = pickle.load(open(key_data_path, 'rb'))
    key_pos = copy.deepcopy(key_data['t_fm_orig'])
    keyframe = key_data['keyframes'][0]
    motion = key_data['motion']
    scene_name = key_data['scene_name']
    speed = np.linalg.norm(np.diff(motion[:, 0, :], axis=0), axis=1)
    distance_to_kf = np.sum(speed[:keyframe])*scale_factor
    distance_to_goal = np.sum(speed[keyframe:])*scale_factor

    navmesh_path = os.path.join(navmesh_dir, scene_name + '_navmesh.ply')
    
    if os.path.exists(navmesh_path):
        navmesh = trimesh.load_mesh(navmesh_path, force='mesh')
    else:
        scene_mesh = trimesh.load_mesh(os.path.join(scene_dir, scene_name + '.ply'), force='mesh')
        scene_mesh_yup = deepcopy(scene_mesh)
        scene_mesh_yup.apply_transform(zup_to_shapenet)
        # scene_mesh_yup.export(prox_folder / (scene_name + '_yup.obj'))
        os.makedirs(navmesh_dir, exist_ok=True)
        navmesh = create_navmesh(scene_mesh_yup, 
                                    export_path=navmesh_path, 
                                    agent_radius=0.1, 
                                    agent_height=1.0, 
                                    agent_max_climb=0.3, 
                                    visualize=False)

    # 面積の最も大きい面を抽出 (要改善)   
    components = trimesh.graph.connected_component_labels(navmesh.face_adjacency)
    components = [np.where(components == i)[0] for i in range(len(np.unique(components)))]
    component_areas = [navmesh.area_faces[c].sum() for c in components]
    largest_component_index = np.argmax(component_areas)
    largest_component_faces = components[largest_component_index]
    largest_mesh = navmesh.submesh([largest_component_faces], append=True)
    navmesh = largest_mesh

    height = key_pos[0][2]
    key_pos[0][2] = np.mean(navmesh.vertices[:,2])
    closest_point, dist, ind = trimesh.proximity.closest_point(navmesh, key_pos)
    # navmesh = get_connected_faces(navmesh, ind)
    # navmesh.export(prox_folder / scene_name / (scene_name + '_navmesh_closest.ply'))

    # key_posがnavmeshの外にあった場合
    if dist > 0.01:
        # closest_pointをnavmeshの内側に少し移動
        vector = closest_point - key_pos
        normalized_vector = vector / np.linalg.norm(vector)
        # closest_point = closest_point + 0.01 * normalized_vector
        normalized_vector = normalized_vector[0]

        # メッシュを拡張
        width = 0.1
        left_vec = rotate_vector(normalized_vector, axis=[0,0,1], angle=-np.pi/2)
        right_vec = rotate_vector(normalized_vector, axis=[0,0,1], angle=np.pi/2)
        l_pos1 = closest_point + left_vec * width
        r_pos1 = closest_point + right_vec * width
        l_pos2 = l_pos1 - normalized_vector * (dist + width)
        r_pos2 = r_pos1 - normalized_vector * (dist + width)
        n_vert = len(navmesh.vertices)
        new_vertices = np.array([l_pos1[0], r_pos1[0], l_pos2[0], r_pos2[0]])
        closest_triangle = navmesh.faces[ind[0]]
        far_ind = np.argmax([np.linalg.norm(key_pos[0] - navmesh.vertices[i]) for i in closest_triangle])
        ind1, ind2 = np.delete(closest_triangle, far_ind)
        far_ind = closest_triangle[far_ind]
        new_faces = [
                    [n_vert, n_vert+1, n_vert+2], 
                    [n_vert+1, n_vert+3, n_vert+2],
                    [n_vert+1, n_vert, far_ind]
                ]
        if np.linalg.norm(navmesh.vertices[ind1] - l_pos1) > np.linalg.norm(navmesh.vertices[ind1] - r_pos1):   # ind1 is close to r_pos1
            new_faces.append([ind1, n_vert+1, far_ind])
            new_faces.append([n_vert, ind2, far_ind])
        else:   # ind2 is close to r_pos1
            new_faces.append([ind2, n_vert+1, far_ind])
            new_faces.append([n_vert, ind1, far_ind])
        all_vertices = np.vstack([navmesh.vertices, new_vertices])
        all_faces = np.vstack([np.delete(navmesh.faces, ind, axis=0), new_faces])
        navmesh = trimesh.Trimesh(vertices=all_vertices, faces=all_faces)

        # 最も近い面のindexの再取得
        closest_point, dist, ind = trimesh.proximity.closest_point(navmesh, key_pos)
        navmesh = get_connected_faces(navmesh, ind)

        navmesh.export(os.path.join(key_data_dir, 'navmesh_pathplanning.ply'))

    else:
        normalized_vector = np.array([1,0,0])


    # define start position
    start = select_circle_point(navmesh, key_pos, distance_to_kf, normalized_vector, n_rot=20)
    goal = select_circle_point(navmesh, key_pos, distance_to_goal, normalized_vector, n_rot=20)

    # path planning
    path_to_kf = path_find(navmesh, start, key_pos, visualize=False)
    path_to_kf = interpolate_path(path_to_kf, speed[:keyframe])
    path_to_goal = path_find(navmesh, key_pos, goal, visualize=False)
    path_to_goal = interpolate_path(path_to_goal, speed[keyframe:])
    path  = np.concatenate([path_to_kf, key_pos, path_to_goal])

    # Path visualization
    spheres = []
    colors = trimesh.visual.interpolate(np.linspace(0, 1, len(path))[:, np.newaxis], 'Reds')
    for i,point in enumerate(path):
        sphere = trimesh.creation.uv_sphere(radius=0.1, count=[20, 20])
        sphere.apply_translation(point)
        if i == keyframe:
            sphere.visual.vertex_colors = np.array([0,255,0,255])
        elif i%2 == 0:
            continue
        else:
            sphere.visual.vertex_colors = colors[i]
        spheres.append(sphere)
    combined_mesh = trimesh.util.concatenate(spheres)
    combined_mesh.export(os.path.join(key_data_dir, 'path.ply'))

    # # flip x
    # path[:, 0] *= -1
    # path -= path[0]

    # height is based on key position
    path[:, 2] = height

    return path

    # save data
    data = {}
    data['global_orient']   = np.zeros([key_data['length'], 3])          
    data['betas']           = np.zeros([key_data['length'], 10])     
    data['body_pose']       = np.zeros([key_data['length'], 63])       
    data['transl']          = path
    data['left_hand_pose']  = np.zeros([key_data['length'], 6])             
    data['right_hand_pose'] = np.zeros([key_data['length'], 6])             
    data['R_fm_orig']       = key_data['R_fm_orig']            
    data['t_fm_orig']       = key_data['t_fm_orig']        
    data['keyframes']       = key_data['keyframes']        
    data['length']          = key_data['length']   
    data['scene_name']      = key_data['scene_name']
    data['key_pose']        = key_data['data']
    save_pkl_path = key_data_path[:-4] + '_path.pkl'
    with open(save_pkl_path, 'wb') as f:
        pickle.dump(data, f)


    # convert to humanml
    POSA_DIR = os.path.dirname(os.path.dirname(__file__))
    neutral_bm_path = os.path.join(POSA_DIR, 'body_models/smplh/neutral/model.npz')
    neutral_dmpl_path = os.path.join(POSA_DIR, 'body_models/dmpls/neutral/model.npz')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    neutral_bm = BodyModel(bm_fname=neutral_bm_path, num_betas=10, num_dmpls=8, dmpl_fname=neutral_dmpl_path).to(device)
    convert_smplx2humanml(
        neutral_bm, 
        device, 
        save_pkl_path, 
        save_pkl_path[:-4] + ".npy", 
        scene_name=scene_name, 
        motion_data=motion,
        flip=False)
    


if __name__ == '__main__':

    scene_dir = "./scenes/"
    navmesh_dir = "./scenes/navmesh/"

    key_data_path = "./save/humanml_only_text_condition/result_a_person_crawls_on_the_floor_and_cleans_/sample12_rep00_iter=20_affordance/pkl/MPH16/072.npy"


    path_planning_prox(
            key_data_path,
            navmesh_dir=navmesh_dir,
            scene_dir=scene_dir,
            scale_factor=0.4)

    print("Done")