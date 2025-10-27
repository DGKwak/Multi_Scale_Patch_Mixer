import os
import numpy as np
import pywt
from pathlib import Path
from skimage import io, filters, transform
from torchvision import datasets
from tqdm import tqdm

def apply_dwt_augmentation(data, wavelet='haar'):
    """
    Apply Discrete Wavelet Transform (DWT) to augment the data.
    Make image to grayscale and apply DWT.
    
    data : image data in numpy array format
    wavelet : type of wavelet to use (default is 'haar')
    """    
    # DWT 적용
    cA, (cH, cV, cD) = pywt.wavedec2(data, wavelet=wavelet, level=1)
    
    Z = np.zeros_like(cA)
    
    # 증강 유형에 따라 3가지 방법으로 증강
    coeff_1 = [cA, (cH, Z, Z)]
    coeff_2 = [cA, (Z, cV, Z)]
    coeff_3 = [cA, (Z, Z, cD)]
    
    # 역 DWT 적용
    aug_1 = pywt.waverec2(coeff_1, wavelet=wavelet)
    aug_2 = pywt.waverec2(coeff_2, wavelet=wavelet)
    aug_3 = pywt.waverec2(coeff_3, wavelet=wavelet)
    
    return aug_1, aug_2, aug_3

def extract_sobel_features(data):
    """
    Extract Sobel features from the image.
    Make Sobel features in three way (horizontal, vertical, diagonal).
    Concatenate the features in channel dimension.
    
    data : image data in numpy array format
    """
    # 데이터를 [0, 1] 범위로 정규화
    if np.max(data) > 1:
        data = data / np.max(data)
    
    # Dx 추출
    Dx = filters.sobel_h(data)
    
    # Dy 추출
    Dy = filters.sobel_v(data)
    
    # Dxy 계산
    Dxy = np.sqrt(np.square(Dx) + np.square(Dy))

    ch_3 = np.stack([Dx, Dy, Dxy], axis=0)
    ch_4 = np.stack([data, Dx, Dy, Dxy], axis=0)
    
    return ch_4, ch_3

if __name__ == "__main__":
    data_path = '/home/eslab/Vscode/MultiPatchdopplerMLP/data/IAA'
    # sobel_path = '/home/eslab/Vscode/MultiPatchdopplerMLP/data/IAA_Sobel'
    sobel_3_path = '/home/eslab/Vscode/MultiPatchdopplerMLP/data/IAA_Sobel_3'
    # dwt_sobel_path = '/home/eslab/Vscode/MultiPatchdopplerMLP/data/IAA_DWT_Sobel'
    dwt_sobel_3_path = '/home/eslab/Vscode/MultiPatchdopplerMLP/data/IAA_DWT_Sobel_3'

    category = ['train', 'val', 'test']

    for cat in category:
        img_data = datasets.ImageFolder(os.path.join(data_path, cat))

        data_sample = list(img_data.samples)
        data_classes = img_data.classes

        for cls in data_classes:
            # if not os.path.exists(os.path.join(sobel_path, cat, cls)):
            #     os.makedirs(os.path.join(sobel_path, cat, cls))
            if not os.path.exists(os.path.join(sobel_3_path, cat, cls)):
                os.makedirs(os.path.join(sobel_3_path, cat, cls))
            # if not os.path.exists(os.path.join(dwt_sobel_path, cat, cls)):
            #     os.makedirs(os.path.join(dwt_sobel_path, cat, cls))
            if not os.path.exists(os.path.join(dwt_sobel_3_path, cat, cls)):
                os.makedirs(os.path.join(dwt_sobel_3_path, cat, cls))

        for img_path, label in tqdm(data_sample, desc=f"Processing {cat}"):
            img_name = Path(img_path).stem

            img = io.imread(img_path, as_gray=True)
            img = transform.resize(img, (224, 224))

            sobel_4_img, sobel_3_img = extract_sobel_features(img)
            dwt_1, dwt_2, dwt_3 = apply_dwt_augmentation(img, wavelet='haar')

            sobel_dwt_4_1, sobel_dwt_3_1 = extract_sobel_features(dwt_1)
            sobel_dwt_4_2, sobel_dwt_3_2 = extract_sobel_features(dwt_2)
            sobel_dwt_4_3, sobel_dwt_3_3 = extract_sobel_features(dwt_3)

            # save_path_sobel = os.path.join(sobel_path, cat, data_classes[label], f'{img_name}_Sobel.npy')
            save_path_sobel_3 = os.path.join(sobel_3_path, cat, data_classes[label], f'{img_name}_Sobel_3.npy')
            # save_path_dwt_1 = os.path.join(dwt_sobel_path, cat, data_classes[label], f'{img_name}_DWT_1.npy')
            # save_path_dwt_2 = os.path.join(dwt_sobel_path, cat, data_classes[label], f'{img_name}_DWT_2.npy')
            # save_path_dwt_3 = os.path.join(dwt_sobel_path, cat, data_classes[label], f'{img_name}_DWT_3.npy')
            save_path_dwt_3_1 = os.path.join(dwt_sobel_3_path, cat, data_classes[label], f'{img_name}_DWT_3_1.npy')
            save_path_dwt_3_2 = os.path.join(dwt_sobel_3_path, cat, data_classes[label], f'{img_name}_DWT_3_2.npy')
            save_path_dwt_3_3 = os.path.join(dwt_sobel_3_path, cat, data_classes[label], f'{img_name}_DWT_3_3.npy')

            # np.save(save_path_sobel, sobel_4_img)
            np.save(save_path_sobel_3, sobel_3_img)
            # np.save(save_path_dwt_1, sobel_dwt_4_1)
            # np.save(save_path_dwt_2, sobel_dwt_4_2)
            # np.save(save_path_dwt_3, sobel_dwt_4_3)
            np.save(save_path_dwt_3_1, sobel_dwt_3_1)
            np.save(save_path_dwt_3_2, sobel_dwt_3_2)
            np.save(save_path_dwt_3_3, sobel_dwt_3_3)