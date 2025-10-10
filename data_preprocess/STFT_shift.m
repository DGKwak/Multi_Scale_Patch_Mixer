%% 
finder = strcat('/media/eslab/Backup/mmWave_dataset/Dataset_848/data/6_February_2019_NG_Homes_Dataset/*.dat');
tmp1 = dir(finder);

cnt = 0;
for j = 1:length(tmp1)
    temp = strrep(tmp1(j).name, '.dat', '');

    % 정규 표현식 패턴
    % 'A' 뒤에 1개 또는 2개(\d{1,2})의 숫자를 찾습니다.
    pattern = 'A(\d{1,2})'; 

    % filename1에서 숫자 추출
    match1 = regexp(temp, pattern, 'tokens');
    result1_num = sprintf('%02d', str2double(match1{1}{1}));
    
    if strcmp(result1_num, '01')
        shift = 100;    
    else
        shift = 50;
    end

    data_name = strcat(tmp1(j).folder, '/', tmp1(j).name);

    shift_STFT_preprocess(data_name, strcat('/home/eslab/Vscode/test_model/data/STFT_shift/', result1_num), temp, shift, cnt)
    cnt = cnt + 1;
end