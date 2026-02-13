import os
import re
import random
import shutil

data_dir = './data/STFT_Split'
labels = os.listdir(data_dir)

save_dir = './data/STFT_PSplit'

# make directories for train, val, test splits
for label in labels:
    for split in ['train', 'val', 'test']:
        dir_path = os.path.join(save_dir, split, label)

        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

regular_pattern = re.compile(r"P\d{2}")
random.seed(42)

for lb in labels:
    pattern_set = set()
    files = os.listdir(os.path.join(data_dir, lb))
    print(f"Processing label: {lb}, Total files: {len(files)}")

    for f in files:
        match = regular_pattern.search(f)
        if match:
            pattern_set.add(match.group(0))

    pattern_list = list(pattern_set)

    random.shuffle(pattern_list)
    print(len(pattern_list))

    train = int(0.7*len(pattern_list))
    val = (len(pattern_list) - train) // 2

    train_patterns = pattern_list[:train]
    val_patterns = pattern_list[train:train+val]
    test_patterns = pattern_list[train+val:]

    patterns = [train_patterns, val_patterns, test_patterns]
    splits = ['train', 'val', 'test']

    for i in range(3):
        file_list = [f for f in files if any(pat in f for pat in patterns[i])]
        
        for f in file_list:
            src_path = os.path.join(data_dir, lb, f)
            dst_path = os.path.join(save_dir, splits[i], lb, f)
            shutil.copy(src_path, dst_path)