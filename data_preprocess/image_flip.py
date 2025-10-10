import os
import torch
import torchvision.transforms.functional as F
from tqdm import tqdm
from PIL import Image

def flip_image(image_path, output_path, image_name):
    image = Image.open(os.path.join(image_path, image_name)).convert("RGB")
    image_tensor = F.to_tensor(image)
    flipped = torch.flip(image_tensor, dims=[1])
    flipped_image = F.to_pil_image(flipped)
    flipped_image.save(os.path.join(output_path, f"flipped_{image_name}"))

def main(input_path, output_path):
    image_list = os.listdir(input_path)

    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    for image_name in tqdm(image_list):
        flip_image(input_path, output_path, image_name)

if __name__ == "__main__":
    main_path = '/home/eslab/Vscode/test_model/data/IAA'
    save_path = '/home/eslab/Vscode/test_model/data/flipped_IAA'
    label = os.listdir(main_path)

    for lbl in label:
        input_path = os.path.join(main_path, lbl)
        output_path = os.path.join(save_path, lbl)

        main(input_path, output_path)