% find folders in Dataset folder
subfolders = dir('/media/eslab/Backup/mmWave_dataset/Dataset_848/data/');
subfolders = subfolders(~ismember({subfolders.name}, {'.', '..'}));

cnt = 0;
% find *.dat files in folders
for i = 1:length(subfolders)
    finder = strcat(subfolders(i).folder, '/', subfolders(i).name, '/6P*.dat');
    tmp1 = dir(finder);
    
    for j = 1:length(tmp1)
        temp = strrep(tmp1(j).name, '.dat', '');
        temp = strrep(temp, '6P', 'x6P');
        
        data_name = strcat(tmp1(j).folder, '/', tmp1(j).name);

        DataProcessingExample_IAA(data_name, '/home/eslab/Vscode/test_model/data/IAA/06', temp, cnt)
        cnt = cnt + 1;
    end
end