import os
import pandas as pd
import glob
import cv2
import numpy as np
import dlib

from tqdm import tqdm
from mtcnn.mtcnn import MTCNN
from collections import defaultdict
from sklearn.model_selection import KFold

def load_data(file_path):
    return pd.read_csv(file_path, header=None)

def save_data(data, file_path):
    data.to_csv(file_path, index=False, header=False)

def fold_division_Track1(train_file, val_file, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    train_df = load_data(train_file)
    val_df = load_data(val_file)

    combined_df = pd.concat([train_df, val_df], ignore_index=True)
    combined_df.columns = ['image', 'valence', 'arousal', 'expression'] + [f'au{i}' for i in range(1, 13)]
    combined_df['video_id'] = combined_df['image'].apply(lambda x: x.split('/')[0])

    combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)

    unique_videos = combined_df['video_id'].unique()

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    for fold, (train_video_idx, test_video_idx) in enumerate(kf.split(unique_videos)):
        train_videos = unique_videos[train_video_idx]
        test_videos = unique_videos[test_video_idx]

        train_fold_df = combined_df[combined_df['video_id'].isin(train_videos)]
        test_fold_df = combined_df[combined_df['video_id'].isin(test_videos)]

        train_fold_df = train_fold_df.sort_values(by='image')
        test_fold_df = test_fold_df.sort_values(by='image')
        train_fold_df = train_fold_df.drop(columns=['video_id'])
        test_fold_df = test_fold_df.drop(columns=['video_id'])

        train_fold_file = f'{output_dir}/train_fold_{fold + 1}.txt'
        train_fold_df.to_csv(train_fold_file, header=False, index=False)

        test_fold_file = f'{output_dir}/val_fold_{fold + 1}.txt'
        test_fold_df.to_csv(test_fold_file, header=False, index=False)

    print("Saving successful!!!")

def _visualize_flow(image_path, flow, save_path):
    hsv = np.zeros_like(cv2.imread(image_path))
    hsv[..., 1] = 255
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    cv2.imwrite(save_path, rgb)


def _Farneback(image_path_list, flow_save_dir, vis_save_dir, visualize=True):
    if len(image_path_list) == 1:
        image_path_list.append(image_path_list[-1])
    prvs = cv2.imread(image_path_list[0], cv2.IMREAD_GRAYSCALE)
    for i in range(1, len(image_path_list)):
        next = cv2.imread(image_path_list[i], cv2.IMREAD_GRAYSCALE)
        prvs = cv2.GaussianBlur(prvs, (5, 5), 0)
        next_frame = cv2.GaussianBlur(next, (5, 5), 0)

        pre_img_index = os.path.basename(image_path_list[i - 1]).split(".")[0]
        last_img_index = os.path.basename(image_path_list[i]).split(".")[0]

        flow_save_path = os.path.join(flow_save_dir, f"{pre_img_index}_{last_img_index}.npy")
        flow = cv2.calcOpticalFlowFarneback(prvs, next_frame, None, 0.5, 4, 15, 5, 5, 1.2, 0) # (112, 112, 2)
        np.save(flow_save_path, flow)

        vis_save_path = os.path.join(vis_save_dir, f"{pre_img_index}_{last_img_index}.png")
        if visualize:
            _visualize_flow(image_path_list[i], flow, vis_save_path)
        prvs = next
    
    if visualize:
        video_name = vis_save_dir.split("/")[-1] + ".mp4"
        video_vis_save_path = os.path.join("/root/code/ABAW/7thDataInfo/data_files/MTL/optical_flow_videos/Farneback", video_name)
        create_video_from_frames(vis_save_dir, video_vis_save_path, fps=15)

def _Lucas_Kanade(image_path_list, flow_save_dir, vis_save_dir, visualize=True):
    prvs_frame = cv2.imread(image_path_list[0])
    prvs_gray = cv2.cvtColor(prvs_frame, cv2.COLOR_BGR2GRAY)
    feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
    p0 = cv2.goodFeaturesToTrack(prvs_gray, mask=None, **feature_params)

    lk_params = dict(winSize=(15, 15), maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    color = np.random.randint(0, 255, (100, 3))

    mask = np.zeros_like(prvs_frame)

    for i in range(1, len(image_path_list)):
        frame = cv2.imread(image_path_list[i])
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        flow, st, err = cv2.calcOpticalFlowPyrLK(prvs_gray, frame_gray, p0, None, **lk_params) # (3, 1, 2)

        if visualize:
            pre_img_index = os.path.basename(image_path_list[i - 1]).split(".")[0]
            last_img_index = os.path.basename(image_path_list[i]).split(".")[0]
            save_path = os.path.join(vis_save_dir, f"{pre_img_index}_{last_img_index}.png")

            good_new = flow[st == 1]
            good_old = p0[st == 1]

            for j, (new, old) in enumerate(zip(good_new, good_old)):
                a, b = new.ravel()
                c, d = old.ravel()
                mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), color[j].tolist(), 2)
                frame = cv2.circle(frame, (int(a), int(b)), 5, color[j].tolist(), -1)

            img = cv2.add(frame, mask)
            cv2.imwrite(save_path, img)
        
        video_name = vis_save_dir.split("/")[-1] + ".mp4"
        video_vis_save_path = os.path.join("/root/code/ABAW/7thDataInfo/data_files/MTL/optical_flow_videos/Farneback", video_name)
        create_video_from_frames(vis_save_dir, video_vis_save_path, fps=15)

def optical_flow_extraction(data_path, method, optical_npy_save_dir, optical_vis_save_dir):
    video_dir_list = os.listdir(data_path)
    for video_dir in video_dir_list:
        video_dir_path = os.path.join(data_path, video_dir)
        vis_save_dir = os.path.join(optical_vis_save_dir, method, video_dir)
        flow_save_dir = os.path.join(optical_npy_save_dir, method, video_dir)
        os.makedirs(vis_save_dir, exist_ok=True)
        os.makedirs(flow_save_dir, exist_ok=True)
        image_path_list = sorted(glob.glob(video_dir_path + "/*.*"))
        if method == "Farneback":
            _Farneback(image_path_list, flow_save_dir, vis_save_dir, visualize=True)
        if method == "Lucas_Kanade":
            _Lucas_Kanade(image_path_list, flow_save_dir, vis_save_dir, visualize=True)
            print("ok")

def _dlib_predictor_extraction(detector, predictor, image_path_list, vis_save_dir, landmark_save_dir):
    for img_path in image_path_list:
        img = cv2.imread(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        results = detector.detect_faces(img)
        for result in results:
            x, y, width, height = result['box']
            keypoints = result['keypoints']
        
            landmarks_points = [
                (keypoints['left_eye'][0], keypoints['left_eye'][1]),
                (keypoints['right_eye'][0], keypoints['right_eye'][1]),
                (keypoints['nose'][0], keypoints['nose'][1]),
                (keypoints['mouth_left'][0], keypoints['mouth_left'][1]),
                (keypoints['mouth_right'][0], keypoints['mouth_right'][1])
            ]

        landmarks_points = np.array(landmarks_points)

        for point in landmarks_points:
            cv2.circle(img, (int(point[0]), int(point[1])), 2, (0, 255, 0), -1)

        landmarks_points = np.array(landmarks_points)
        landmarks_npy_path = os.path.join(landmark_save_dir, img_path.split('.')[0] + '.npy')
        np.save(landmarks_npy_path, landmarks_points)

        vis_img_path = os.path.join(vis_save_dir, img_path.split('.')[0] + '.png')
        cv2.imwrite(vis_img_path, img)

def facial_landmark_extraction(data_path, landmark_npy_save_dir, landmark_vis_save_dir):
    video_dir_list = os.listdir(data_path)
    detector = MTCNN()
    predictor = dlib.shape_predictor("/project/liuchen/pth/shape_predictor_68_face_landmarks_GTX.dat")
    for video_dir in video_dir_list:
        video_dir_path = os.path.join(data_path, video_dir)

        vis_save_dir = os.path.join(landmark_vis_save_dir, video_dir)
        landmark_save_dir = os.path.join(landmark_npy_save_dir, video_dir)
        os.makedirs(vis_save_dir, exist_ok=True)
        os.makedirs(landmark_save_dir, exist_ok=True)
        
        image_path_list = sorted(glob.glob(video_dir_path + "/*.png"))
        _dlib_predictor_extraction(detector, predictor, image_path_list, vis_save_dir, landmark_save_dir)
    
        video_name = vis_save_dir.split("/")[-1] + ".mp4"
        video_vis_save_path = os.path.join("/root/code/ABAW/7thDataInfo/data_files/MTL/facial_landmark_videos", video_name)
        create_video_from_frames(vis_save_dir, video_vis_save_path, fps=15)
        
def create_video_from_frames(input_folder, output_video_path, fps=15):
    images = sorted([img for img in sorted(glob.glob(input_folder + "/*.png"))])
    frame = cv2.imread(os.path.join(input_folder, images[0]))
    height, width, layers = frame.shape
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    for image in images:
        frame = cv2.imread(image)
        video.write(frame)
    
    video.release()
    
def drop_duplicated_lines_from_txt(file_path, new_save_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    lines = sorted(set(lines))
    with open(new_save_path, 'w') as file:
        file.writelines(lines)

def drop_duplicated_lines_from_csv(file_path, new_save_path):
    df = pd.read_csv(file_path, header=None)
    df = df.drop_duplicates()
    df.to_csv(new_save_path, index=False)

def get_frame_counts(root_dir):
    frame_counts = {}
    folder_list = os.listdir(root_dir)
    for folder in folder_list:
        print(folder)
        folder_path = os.path.join(root_dir, folder)
        if os.path.isdir(folder_path):
            frame_counts[folder] = len(os.listdir(folder_path))
    return frame_counts

affwild_label_dict = {0: "Neutral", 1: "Anger", 2: "Disgust", 3: "Fear", 4: "Happiness", 5: "Sadness", 6: "Surprise"}
raf_db_label_dict = {1: "Surprise", 2: "Fear", 3: "Disgust ", 4: "Happiness", 5: "Sadness", 6: "Anger", 7: "Neutral"}

def _write_list_txt(file_path, data_list, label_dict, dataset_name=None):
    with open(file_path, 'w') as file:
        for item in data_list:
            # file.write('%s\n' % item) 
            image_index = int(item.split("/")[-1].split(".")[0])
            v_name = item.split("/")[0]
            if image_index >= len(label_dict[v_name]):
                continue
            label = label_dict[v_name][image_index]
            if int(label) == -1:
                continue
            if dataset_name == "affwild":
                if int(label) == 0 or int(label) == 7:
                    continue
                label_txt = affwild_label_dict[int(label)] + "\n"
            if dataset_name == "raf":
                label_txt = raf_db_label_dict[int(label)] + "\n"
            if dataset_name is None:
                label_txt = label
            file.write(item.strip() + ',' + label_txt)
    file.close()

def _read_data_txt(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    if not lines[0].isdigit():
        lines = lines[1:]
    return lines

def fold_division_Track2(frame_counts, datafile_save_path, label_txt_dir, n_splits=5):
    track_dir2 = "/project/ABAW6/data/challenge4/crop_face/"

    folders = list(frame_counts.keys())
    frame_numbers = list(frame_counts.values())
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    train_folds = defaultdict(list)
    val_folds = defaultdict(list)

    for fold_idx, (train_index, val_index) in enumerate(kf.split(folders, frame_numbers)):
        for idx in train_index:
            train_folds[fold_idx].append(folders[idx])
        
        for idx in val_index:
            val_folds[fold_idx].append(folders[idx])

    # Reading labels from txt file
    for i in range(n_splits):
        print(f"fold {i}")
        train_video_list = train_folds[i]
        val_video_list = val_folds[i]

        fold_train_image_list = []
        train_vlabel_dict = {}
        for v_name in train_video_list:
            v_path = os.path.join(track_dir2, v_name)
            v_label_path = os.path.join(label_txt_dir, v_name + ".txt")
            anno_lines = _read_data_txt(v_label_path)
            train_vlabel_dict[v_name] = anno_lines
            fold_train_image_list.extend([f"{v_name}/{img_name}" for img_name in sorted(os.listdir(v_path))])
        
        fold_val_image_list = []
        val_vlabel_dict = {}
        for v_name in val_video_list:
            v_path = os.path.join(track_dir2, v_name)
            v_label_path = os.path.join(label_txt_dir, v_name + ".txt")
            anno_lines = _read_data_txt(v_label_path)
            val_vlabel_dict[v_name] = anno_lines
            fold_val_image_list.extend([f"{v_name}/{img_name}" for img_name in sorted(os.listdir(v_path))])
        
        f_train_save_path = f"{datafile_save_path}/train_fold_{i + 1}.txt"
        f_val_save_path = f"{datafile_save_path}/val_fold_{i + 1}.txt"

        _write_list_txt(f_train_save_path, fold_train_image_list, train_vlabel_dict)
        _write_list_txt(f_val_save_path, fold_val_image_list, val_vlabel_dict)
    
    return train_folds, val_folds

def _get_video_names(set_dir):
    v_names = [v_name.split(".")[0] for v_name in os.listdir(set_dir)]
    return v_names

def AffWild_datafile_gen(trainset_dir, valset_dir, data_dir):
    train_vnames = _get_video_names(trainset_dir)
    val_vnames = _get_video_names(valset_dir)

    train_data_list = []
    val_data_list = []
    train_vlabel_dict = {}
    val_vlabel_dict = {}

    for v_name in tqdm(train_vnames):
        v_label_path = os.path.join(trainset_dir, v_name + ".txt")
        v_img_dir = os.path.join(data_dir, v_name)
        if not os.path.exists(v_img_dir):
            continue

        v_labels = _read_data_txt(v_label_path)

        v_img_names = [f"{v_name}/{img_name}" for img_name in sorted(os.listdir(v_img_dir))]
        train_data_list.extend(v_img_names)
        if len(v_labels) != len(v_img_names):
            print(f"{v_name} label_number/image_number: {len(v_labels)}/{len(v_img_names)}")
        train_vlabel_dict[v_name] = v_labels

    for v_name in val_vnames:
        v_label_path = os.path.join(valset_dir, v_name + ".txt")
        v_img_dir = os.path.join(data_dir, v_name)
        v_labels = _read_data_txt(v_label_path)

        v_img_names = [f"{v_name}/{img_name}" for img_name in sorted(os.listdir(v_img_dir))]
        val_data_list.extend(v_img_names)
        if len(v_labels) != len(v_img_names):
            print(f"{v_name} label_number/image_number: {len(v_labels)}/{len(v_img_names)}")
        val_vlabel_dict[v_name] = v_labels
    
    _write_list_txt("/root/code/ABAW/7thDataInfo/Aff-wild2/train.txt", train_data_list, train_vlabel_dict, dataset_name="affwild")
    _write_list_txt("/root/code/ABAW/7thDataInfo/Aff-wild2/val.txt", val_data_list, val_vlabel_dict, dataset_name="affwild")
        
def raf_single_datafile_gen(raf_single_data, data_dir, split="test", save_path="/root/code/ABAW/7thDataInfo/RAF-DB/val_single.txt"):
    with open(raf_single_data, "r") as file:
        lines = file.readlines()
    
    with open(save_path, 'w') as file:
        for line in lines:
            img_name = line.split(" ")[0]
            if split not in img_name:
                continue
            label = int(line.split(" ")[-1])
            if label == 7:
                continue

            img_path = os.path.join(data_dir, img_name.replace(".jpg", "_aligned.jpg"))
            if not os.path.exists(img_path):
                print(print(f"{img_path} does not exist!!!!"))

            file.write(img_name.replace(".jpg", "_aligned.jpg") + ',' + raf_db_label_dict[label] + '\n')

def raf_compound_datafile_gen(data_dir, split="train", save_path="/root/code/ABAW/7thDataInfo/RAF-DB/train_compound.txt"):
    img_list = glob.glob(data_dir + "/*/*.*")
    with open(save_path, 'w') as file:
        for img_path in sorted(img_list):
            if split not in img_path:
                continue
            label = img_path.split("/")[-2]
            if label not in ["Angrily Surprised", "Disgustedly Surprised", "Fearfully Surprised", "Happily Surprised", "Sadly Angry", "Sadly Fearful", "Sadly Surprised"]:
                continue
            label = label.replace("Happily", "Happiness").replace("Disgustly", "Disgust").replace("Disgusted", "Disgust").replace("Disgustedly", "Disgust").replace("Sadly", "Sadness").replace("Surprised", "Surprise").replace("Angrily", "Anger").replace("Angry", "Anger").replace("Fearfully", "Fear").replace("Fearful", "Fear")
            image_name = os.path.basename(img_path)
            file.write(img_path.split("/")[-2] + "/" + image_name.strip() + ',' + label + '\n')

if __name__ == "__main__":
    # train_file_ori_txt = '/root/code/ABAW/7thDataInfo/training_set_annotations_ori.txt'
    # val_file_ori_txt = '/root/code/ABAW/7thDataInfo/validation_set_annotations_ori.txt'
    # train_file_new_txt = '/root/code/ABAW/7thDataInfo/training_set_annotations.txt'
    # val_file_new_txt = '/root/code/ABAW/7thDataInfo/validation_set_annotations.txt' 
    # drop_duplicated_lines_from_txt(train_file_ori_txt, train_file_new_txt)
    # drop_duplicated_lines_from_txt(val_file_ori_txt, val_file_new_txt)

    # train_file_ori_csv = '/root/code/ABAW/dataload/train_ori.csv'
    # val_file_ori_csv = '/root/code/ABAW/dataload/val_ori.csv'
    # train_file_new_csv = '/root/code/ABAW/dataload/train.csv'
    # val_file_new_csv = '/root/code/ABAW/dataload/val.csv'
    # drop_duplicated_lines_from_csv(train_file_ori_csv, train_file_new_csv)
    # drop_duplicated_lines_from_csv(val_file_ori_csv, val_file_new_csv)

    # output_dir = '/root/code/ABAW/7thDataInfo/five_fold'

    # fold_division(train_file_new_csv, val_file_new_csv, output_dir)

    # data_path = "/root/code/ABAW/7thDataInfo/data_files/MTL/cropped_aligned"
    # method = "Farneback"
    # method = "Lucas_Kanade"
    # optical_npy_save_dir = "/root/code/ABAW/7thDataInfo/data_files/MTL/optical_flow"
    # optical_vis_save_dir = "/root/code/ABAW/7thDataInfo/data_files/MTL/optical_flow_visualize"
    # optical_flow_extraction(data_path, method, optical_npy_save_dir, optical_vis_save_dir)

    # frame_dir = "/root/code/ABAW/7thDataInfo/data_files/MTL/optical_flow/Farneback/446"
    # output_video_path = "/root/code/ABAW/7thDataInfo/data_files/MTL/optical_flow_videos/426.mp4"
    # create_video_from_frames(frame_dir, output_video_path, fps=15)

    # landmark_npy_save_dir = "/root/code/ABAW/7thDataInfo/data_files/MTL/facial_landmarks"
    # landmark_vis_save_dir = "/root/code/ABAW/7thDataInfo/data_files/MTL/facial_landmarks_visualize"
    # facial_landmark_extraction(data_path, landmark_npy_save_dir, landmark_vis_save_dir)

    # track_dir2 = "/project/ABAW6/data/challenge4/crop_face/"
    # frame_counts = get_frame_counts(track_dir2)

    # data_save_dir = "/root/code/ABAW/7thDataInfo/Track2_five_fold"
    # label_txt_dir = "/project/ABAW6/data/challenge4/annos_singleframe_qf/"

    # fold_division_Track2(frame_counts, data_save_dir, label_txt_dir, n_splits=5)

    # trainset_dir = "/project/ABAW6/data/challenge1_3/Annotations/EXPR_Recognition_Challenge/Train_Set/"
    # valset_dir = "/project/ABAW6/data/challenge1_3/Annotations/EXPR_Recognition_Challenge/Validation_Set/"
    # data_dir = "/project/ABAW6/data/challenge1_3/crop_face_retinaface2/"

    # AffWild_datafile_gen(trainset_dir, valset_dir, data_dir)

    raf_single_data = "/project/ABAW6/data/RAF_single/list_patition_label.txt"
    raf_single_data_dir = "/project/ABAW6/data/RAF_single/aligned/"
    raf_single_datafile_gen(raf_single_data, raf_single_data_dir, split="test", save_path="/root/code/ABAW/7thDataInfo/data_files/RAF-DB/val_single.txt")
    raf_single_datafile_gen(raf_single_data, raf_single_data_dir, split="train", save_path="/root/code/ABAW/7thDataInfo/data_files/RAF-DB/train_single.txt")


    raf_compound_dir = "/project/ABAW6/data/RAF_compound/images/"
    raf_compound_datafile_gen(raf_compound_dir, split="test", save_path="/root/code/ABAW/7thDataInfo/data_files/RAF-DB/val_compound_clean.txt")
    raf_compound_datafile_gen(raf_compound_dir, split="train", save_path="/root/code/ABAW/7thDataInfo/data_files/RAF-DB/train_compound_clean.txt")
