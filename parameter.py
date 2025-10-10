from Reference_code.Hire_MLP import hire_mlp_tiny
from Reference_code.wave_mlp import WaveMLP_T

H_model = hire_mlp_tiny()
W_model = WaveMLP_T()

H_params = sum(p.numel() for p in H_model.parameters() if p.requires_grad)
W_params = sum(p.numel() for p in W_model.parameters() if p.requires_grad)

print(f'HireMLP parameters: {H_params}')
print(f'WaveMLP parameters: {W_params}')