import os
import glob
import random
import shutil
from tqdm import tqdm

def output_path_mkdir(output_dir: str, subfolder: list):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")
    
    for subdir in ['train', 'val', 'test']:
        subdir_path = os.path.join(output_dir, subdir)
        
        if not os.path.exists(subdir_path):
            os.makedirs(subdir_path)
            print(f"Created directory: {subdir_path}")

        for folder in subfolder:
            folder_path = os.path.join(subdir_path, folder)

            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
                print(f"Created directory: {folder_path}")

def split_file(data_dir: str, output_dir: str, subfolder: list, data_split: list):
    train_dict = {}
    val_dict = {}
    test_dict = {}
    
    for f in subfolder:
        path = os.path.join(data_dir, f)

        data = glob.glob(os.path.join(path, '*.png'))
        random.shuffle(data)

        data_len = len(data)
        print(f"Total files in {f}: {data_len}")
        train_end = int(data_len * data_split[0])
        val_end = train_end + int(data_len * data_split[1])

        train_data = data[:train_end]
        val_data = data[train_end:val_end]
        test_data = data[val_end:]

        train_dict[f] = train_data
        val_dict[f] = val_data
        test_dict[f] = test_data

    return train_dict, val_dict, test_dict

def copy_files(output_dir: str, label: str, train: list, val: list, test: list):
    for file in tqdm(train, desc=f"Copying training files for {label}"):
        shutil.copy(file, os.path.join(output_dir, 'train', label))

    for file in tqdm(val, desc=f"Copying validation files for {label}"):
        shutil.copy(file, os.path.join(output_dir, 'val', label))

    for file in tqdm(test, desc=f"Copying test files for {label}"):
        shutil.copy(file, os.path.join(output_dir, 'test', label))

def main():
    data_dir = '/home/eslab/Vscode/test_model/data/data_split_test'
    output_dir = '/home/eslab/Vscode/test_model/data/data_split_output'
    
    split_ratios = [0.7, 0.2]

    folders = os.listdir(data_dir)

    output_path_mkdir(output_dir, folders)
    train, val, test = split_file(data_dir, output_dir, folders, split_ratios)
    
    for lb in list(train.keys()):
        print(f"Processing label: {lb}")
        copy_files(output_dir, lb, train[lb], val[lb], test[lb])

if __name__ == "__main__":
    main()