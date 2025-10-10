%==========================================================================
% Author Abdullah AKAYDIN
% Version 1.0
%==========================================================================

function [doppler_profiles] = IAA_Spectogram(Data, A, vFreqVel)

% global vFreqVel
doppler_profiles = zeros(1,length(vFreqVel));

for iaa = 1:1
ref_coef = zeros(1,length(A(1,:)));
% initial delay-sum estimation
    for ee = 1:length(A(1,:))
    ref_coef(ee) = (A(:,ee)'*Data.')/(A(:,ee)'*A(:,ee));
    end
%  Iterative loop to refine the reflection coefficient (can be 8 or 10 or smaller number)
for iia_loop = 1:10
R_iaa = 0;
% Data covariance matrix update
P = zeros(256);
P = diag(abs(ref_coef).^2);
R_iaa = (A*P*A');
%  Data covariance matrix inverse
R_inv=inv(R_iaa);
% Reflection coefficient calculation of the each array steering vectors (from -90 to 90)
for ccc = 1:length(A(1,:))
    ref_coef(ccc) = (A(:,ccc)'*(R_inv)*(Data.'))/(A(:,ccc)'*(R_inv)*A(:,ccc));
end
end
doppler_profiles(iaa,:)= ref_coef;
end