function Data_aug = Resampling(Data_spec, scale_factor)
    [H, W] = size(Data_spec); 
    H_new = round(H * scale_factor);
    
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
end