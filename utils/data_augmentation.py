import numpy as np
import pywt
from skimage import filters, transform

def apply_dwt_augmentation(data, aug_type, wavelet='haar', image_size=(224, 224)):
    """
    Apply Discrete Wavelet Transform (DWT) to augment the data.
    Make image to grayscale and apply DWT.
    
    data : image data in numpy array format
    aug_type : type of augmentation (0 : no augmentation, 1 : cA + cH, 2 : cA + cV, 3 : cA + cD)
    wavelet : type of wavelet to use (default is 'haar')
    image_size : size of the output image (default is (224, 224))
    """
    # 증강하지 않는 경우 원본 이미지 리사이즈 후 return
    if not aug_type:
        return transform.resize(data, image_size)
    
    # DWT 적용
    cA, (cH, cV, cD) = pywt.wavedec2(data, wavelet=wavelet, level=1)
    
    Z = np.zeros_like(cA)
    
    # 증강 유형에 따라 3가지 방법으로 증강
    if aug_type == 1:
        coeff = [cA, (cH, Z, Z)]
    elif aug_type == 2:
        coeff = [cA, (Z, cV, Z)]
    elif aug_type == 3:
        coeff = [cA, (Z, Z, cD)]
    
    # 역 DWT 적용
    augmented_data = pywt.waverec2(coeff, wavelet=wavelet)
    
    return transform.resize(augmented_data, image_size)

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
    
    return Dx, Dy, Dxy