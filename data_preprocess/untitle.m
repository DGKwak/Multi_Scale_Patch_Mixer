% find folders in Dataset folder
subfolders = dir('/media/eslab/Backup/mmWave_dataset/Dataset_848/data/');
subfolders = subfolders(~ismember({subfolders.name}, {'.', '..'}));

finder = strcat(subfolders(1).folder, '/', subfolders(1).name, '/*.dat');
tmp1 = dir(finder);

filename = strcat(tmp1(1).folder, '/', tmp1(1).name);

Augment(filename, './', 'test', '1')