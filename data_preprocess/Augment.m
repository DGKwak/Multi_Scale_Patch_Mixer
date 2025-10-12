%% Creates a quick look of Range-Time and Spectrograms from a data file
%==========================================================================
% Author UoG Radar Group
% Version 1.0

% The user has to select manually the range bins over which the
% spectrograms are calculated. There may be different ways to calculate the
% spectrogram (e.g. coherent sum of range bins prior to STFT). 
% Note that the scripts have to be in the same folder where the data file
% is located, otherwise uigetfile() and textscan() give an error. The user
% may replace those functions with manual read to the file path of a
% specific data file
%==========================================================================
%% Data reading part
%[filename,pathname] = uigetfile('*.dat');
function Augment(filename, save_folder, save_file, cnt)
fileID = fopen(filename, 'r');
dataArray = textscan(fileID, '%f');
fclose(fileID);
radarData = dataArray{1};
clearvars fileID dataArray ans;
fc = radarData(1); % Center frequency
Tsweep = radarData(2); % Sweep time in ms
Tsweep=Tsweep/1000; %then in sec
NTS = radarData(3); % Number of time samples per sweep
Bw = radarData(4); % FMCW Bandwidth. For FSK, it is frequency step;
% For CW, it is 0.
Data = radarData(5:end); % raw data in I+j*Q format
fs=NTS/Tsweep; % sampling frequency ADC
record_length=length(Data)/NTS*Tsweep; % length of recording in s
nc=record_length/Tsweep; % number of chirps

%% Reshape data into chirps and plot Range-Time
Data_time=reshape(Data, [NTS nc]);
win = ones(NTS,size(Data_time,2));
%Part taken from Ancortek code for FFT and IIR filtering
tmp = fftshift(fft(Data_time.*win),1);
Data_range(1:NTS/2,:) = tmp(NTS/2+1:NTS,:);
ns = oddnumber(size(Data_range,2))-1;
Data_range_MTI = zeros(size(Data_range,1),ns);
[b,a] = butter(4, 0.0075, 'high');
[h, f1] = freqz(b, a, ns);
for k=1:size(Data_range,1)
  Data_range_MTI(k,1:ns) = filter(b,a,Data_range(k,1:ns));
end
freq =(0:ns-1)*fs/(2*ns); 
range_axis=(freq*3e8*Tsweep)/(2*Bw);
Data_range_MTI=Data_range_MTI(2:size(Data_range_MTI,1),:);
Data_range=Data_range(2:size(Data_range,1),:);

%% Spectrogram processing for 2nd FFT to get Doppler
% This selects the range bins where we want to calculate the spectrogram
bin_indl = 10;
bin_indu = 30;

MD.PRF=1/Tsweep;
MD.TimeWindowLength = 200;
MD.OverlapFactor = 0.95;
MD.OverlapLength = round(MD.TimeWindowLength*MD.OverlapFactor);
MD.Pad_Factor = 4;
MD.FFTPoints = MD.Pad_Factor*MD.TimeWindowLength;
MD.DopplerBin=MD.PRF/(MD.FFTPoints);
MD.DopplerAxis=-MD.PRF/2:MD.DopplerBin:MD.PRF/2-MD.DopplerBin;
MD.WholeDuration=size(Data_range_MTI,2)/MD.PRF;
MD.NumSegments=floor((size(Data_range_MTI,2)-MD.TimeWindowLength)/floor(MD.TimeWindowLength*(1-MD.OverlapFactor)));
    
Data_spec_MTI2=0;
Data_spec2=0;
for RBin=bin_indl:1:bin_indu
    Data_MTI_temp = fftshift(spectrogram(Data_range_MTI(RBin,:),MD.TimeWindowLength,MD.OverlapLength,MD.FFTPoints),1);
    Data_spec_MTI2=Data_spec_MTI2+abs(Data_MTI_temp);                                
    Data_temp = fftshift(spectrogram(Data_range(RBin,:),MD.TimeWindowLength,MD.OverlapLength,MD.FFTPoints),1);
    Data_spec2=Data_spec2+abs(Data_temp);
end
MD.TimeAxis=linspace(0,MD.WholeDuration,size(Data_spec_MTI2,2));

Data_spec_MTI2=flipud(Data_spec_MTI2);

%% ------------------- 주파수 스케일링 증강 적용 -------------------
gamma_f = 0.7;
Data_spec = Data_spec_MTI2;

[H, W] = size(Data_spec); 
H_new = round(H * gamma_f);
    
Data_aug = zeros(H, W, 'like', Data_spec); % 최종 행렬 초기화 (원본 데이터형 유지)
    
% 원본 주파수 축의 인덱스 (1부터 H까지)
x_orig = linspace(1, H, H); 
    
% 1. H -> H_new (확대/축소)에 사용할 인덱스 생성
x_scaled = linspace(1, H, H_new); 
    
% 2. H_new -> H (원래 크기로 복원)에 사용할 인덱스 생성
x_final = linspace(1, H_new, H); 
    
for w = 1:W % 모든 시간 축(열)에 대해 반복
    % 1. H -> H_new: 각 열(신호)을 새로운 크기로 리샘플링
    % Data_spec(:, w)는 800개의 데이터 포인트를 가짐. 이를 H_new 포인트로 변환.
    col_scaled = interp1(x_orig, Data_spec(:, w), x_scaled, 'linear', 'extrap'); 
    
    % 2. H_new -> H: 리샘플링된 신호를 원래 H 크기로 복원
    % col_scaled는 H_new 포인트를 가짐. 이를 다시 800 포인트로 변환.
    Data_aug(:, w) = interp1(linspace(1, H_new, H_new), col_scaled, x_final, 'linear', 'extrap');
end

data = Data_aug;

%% -------------------------------------------------------------

hdl_fig = figure('Position',[0,0,800,800], 'Visible', 'off');
set(gca, 'Position', [0,0,1,1]);
set(gcf, 'PaperPositionMode', 'auto');
figure(hdl_fig); colormap 'jet';
imagesc(20*log10(abs(data)));
axis off;
clim = get(gca, "Clim");
set(gca, 'CLim', clim(2)+[-50,0]);

save_path = strcat(save_folder, '/augmented_', save_file, '_', sprintf('%05d', cnt),'.png');
print(save_path, '-dpng', '-r300')
close(hdl_fig);

end