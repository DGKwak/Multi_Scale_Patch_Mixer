import os
import glob
import numpy as np
import torch
from torchvision import datasets
from torch.utils.data import Dataset
from skimage import io

# from utils.augmentation import apply_dwt_augmentation, extract_sobel_features

# class MD_dataset(Dataset):
#     def __init__(self, 
#                  path, 
#                  is_train=True, 
#                  image_size=(224,224)):
#         self.is_train = is_train
#         self.image_size = image_size
        
#         self.data = datasets.ImageFolder(root=path)
#         self.file_list = list(self.data.samples)
        
#         # Training 시에는 idx 4배 증강, 평가 시에는 원본만 사용
#         self.num_augmentations = 4 if is_train else 1
    
#     def get_class_names(self):
#         return list(self.data.classes)
    
#     def __len__(self):
#         return len(self.file_list) * self.num_augmentations
    
#     def __getitem__(self, idx):
#         # 인덱스 매핑
#         original_idx = idx // self.num_augmentations
        
#         # 0: 원본, 1: cA+cH, 2: cA+cV, 3: cA+cD
#         aug_type = idx % self.num_augmentations
        
#         filepath, label = self.file_list[original_idx]
        
#         # 이미지 로드
#         img = io.imread(filepath, as_gray=True)
        
#         # DWT 증강 적용
#         img_dwt = apply_dwt_augmentation(img, aug_type, image_size=self.image_size)
        
#         # Sobel 특징 추출
#         Dx, Dy, Dxy = extract_sobel_features(img_dwt)
        
#         # 채널 차원으로 결합
#         input_numpy = np.stack([Dx, Dy, Dxy], axis=0)
        
#         # numpy 배열을 torch 텐서로 변환
#         input_tensor = torch.from_numpy(input_numpy).float()
#         label_tensor = torch.tensor(label, dtype=torch.long)

#         return input_tensor, label_tensor

class Sobel_dataset(Dataset):
    def __init__(self,
                 data_path,
                 is_train=True,):
        self.data_path = data_path
        self.is_train = is_train
        
        self.class_to_idx = {}
        self.data_list = []
        
        for class_idx, class_name in enumerate(sorted(os.listdir(self.data_path))):
            class_path = os.path.join(self.data_path, class_name)            
            self.class_to_idx[class_name] = class_idx
            
            for file in glob.glob(os.path.join(class_path, '*.npy')):
                self.data_list.append((file, class_idx))
        
        self.label_list = [label for _, label in self.data_list]

    def get_classes(self):
        return sorted(os.listdir(self.data_path))

    def __len__(self):
        return len(self.data_list)
    
    def get_labels(self):
        return self.label_list

    def __getitem__(self, idx):
        file_path, label = self.data_list[idx]
        
        sample_np = np.load(file_path)
        sample_np = sample_np.astype(np.float32, copy=True)
        
        sample = torch.from_numpy(sample_np)
        label = torch.tensor(label, dtype=torch.long)
        
        return sample, label