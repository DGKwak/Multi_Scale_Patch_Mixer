%% Creates a quick look of Range-Time and Spectrograms from a data file
%==========================================================================
% Author Abdullah AKAYDIN
% Version 1.0

% The user has to select manually the range bins over which the
% spectrograms are calculated. There may be different ways to calculate the
% spectrogram (e.g. coherent sum of range bins prior to Iterative adaptive Approach(IAA)). 
% Note that the scripts have to be in the same folder where the data file
% is located, otherwise uigetfile() and textscan() give an error. The user
% may replace those functions with manual read to the file path of a
% specific data file. IAA_Spectogram() function calculates the IAA Doppler
% spectrum.
%==========================================================================
%% Data reading part
function DataProcessingExample_IAA(filename, save_folder, save_file, cnt)
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
%Parameters for spectrograms
MD.PRF=1/Tsweep;
MD.TimeWindowLength = 100;

MD.OverlapFactor = 0.95;
MD.OverlapLength = round(MD.TimeWindowLength*MD.OverlapFactor);
MD.IAAPoints = 800;
MD.DopplerBin=MD.PRF/(MD.IAAPoints);
MD.DopplerAxis=-MD.PRF/2:MD.DopplerBin:MD.PRF/2-MD.DopplerBin;
MD.WholeDuration=size(Data_range_MTI,2)/MD.PRF;
MD.NumSegments=floor((size(Data_range_MTI,2)-MD.TimeWindowLength)/floor(MD.TimeWindowLength*(1-MD.OverlapFactor)));
    
Data_spec_MTI2=0;
Data_spec2=0;
indx = 1:MD.TimeWindowLength-MD.OverlapLength:nc-MD.TimeWindowLength+1;

vFreqVel =  -500:MD.DopplerBin:500-MD.DopplerBin; % super-resolution Doppler grid
vVel = vFreqVel.*3e8/2/5.8e9;
chirp= 0:MD.TimeWindowLength-1;
for aa = 1:length(vFreqVel)
A(:,aa)= exp(-1i.*2.*(pi).*(chirp').*(vFreqVel(aa)).*Tsweep);
end
tic
for RBin = bin_indl:bin_indu
    Data_MTI_temp = zeros(length(vFreqVel), length(indx));
    for k = 1:length(indx)
        i = (k - 1) * (MD.TimeWindowLength-MD.OverlapLength) + 1;
        % Call IAA_Spectogram
        Data_MTI_temp(:, k) = Data_MTI_temp(:,k) + IAA_Spectogram(Data_range_MTI(RBin, i:i+MD.TimeWindowLength-1), A,vFreqVel).';
    end
    Data_spec_MTI2 = Data_spec_MTI2 + Data_MTI_temp;
end
toc
    MD.TimeAxis=linspace(0,MD.WholeDuration,size(Data_spec_MTI2,2));


hdl_fig = figure('Position',[0,0,800,800], 'Visible', 'off');
set(gca, 'Position', [0,0,1,1]);
set(gcf, 'PaperPositionMode', 'auto');

figure(hdl_fig); colormap 'jet';
imagesc(MD.TimeAxis,MD.DopplerAxis.*3e8/2/5.8e9,20*log10(abs(Data_spec_MTI2)));
axis off;

clim = get(gca, "Clim");
set(gca, 'CLim', clim(2)+[-50,0]);

save_path = strcat(save_folder, '/' ,save_file, '_', sprintf('%05d', cnt),'.png');
print(save_path, '-dpng', '-r300')
close(hdl_fig);

end