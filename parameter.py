import model.Multi_Scale_Patch_Mixer_with_Shift as Shift
import model.Multi_Scale_Patch_Mixer_with_Shift_Squeezed_hidden as SQShift
import model.Lightweight_Multi_Scale_Patch_Mixer_with_Shift as LWShift
from Reference_code.MLP_Mixer import MLPMixer

# Shift 모델 기본 기준
# Shift_768_2_8 = Shift.MultiscaleMixer(3, 768, 8, 0.1, patches=[(224, 2), (224, 4)], act='relu')
# Shift_128_2_8 = Shift.MultiscaleMixer(3, 128, 8, 0.1, patches=[(224, 2), (224, 4)], act='relu')
# Shift_768_1_8 = Shift.MultiscaleMixer(3, 768, 8, 0.1, patches=[(224, 2)], act='relu')
# Shift_128_1_8 = Shift.MultiscaleMixer(3, 128, 8, 0.1, patches=[(224, 2)], act='relu')

# Hidden dimension 1.5배로 줄인 Shift 모델 기준
# SQShift_768_2_8 = SQShift.MultiscaleMixer(3, 768, 8, 0.1, patches=[(224, 2), (224, 4)], act='relu')
# SQShift_128_2_8 = SQShift.MultiscaleMixer(3, 128, 8, 0.1, patches=[(224, 2), (224, 4)], act='relu')
# SQShift_768_1_8 = SQShift.MultiscaleMixer(3, 768, 8, 0.1, patches=[(224, 2)], act='relu')
# SQShift_128_1_8 = SQShift.MultiscaleMixer(3, 128, 8, 0.1, patches=[(224, 2)], act='relu')

# 패치 임베딩에 DSC 적용한 Shift 모델 기준
# LWShift_768_2_8 = LWShift.MultiscaleMixer(3, 768, 8, 0.1, patches=[(224, 2), (224, 4)], act='relu')
# LWShift_128_2_8 = LWShift.MultiscaleMixer(3, 128, 8, 0.1, patches=[(224, 2), (224, 4)], act='relu')
# LWShift_768_1_8 = LWShift.MultiscaleMixer(3, 768, 8, 0.1, patches=[(224, 2)], act='relu')
# LWShift_128_1_8 = LWShift.MultiscaleMixer(3, 128, 8, 0.1, patches=[(224, 2)], act='relu')

save_path = './results/model_param/'

# file_name = ['Shift_768_2_8', 'Shift_768_1_8', 'Shift_128_2_8', 'Shift_128_1_8',
#              'SHShift_768_2_8', 'SHShift_768_1_8', 'SHShift_128_2_8', 'SHShift_128_1_8',
#              'LWShift_768_2_8', 'LWShift_768_1_8', 'LWShift_128_2_8', 'LWShift_128_1_8']

# models = {'Shift_768_2_8': [Shift_768_2_8, 768, [(224, 2), (224, 4)], 8, 'relu', 0.1],
#           'Shift_768_1_8': [Shift_768_1_8, 768, [(224, 2)], 8, 'relu', 0.1],
#           'Shift_128_2_8': [Shift_128_2_8, 128, [(224, 2), (224, 4)], 8, 'relu', 0.1],
#           'Shift_128_1_8': [Shift_128_1_8, 128, [(224, 2)], 8, 'relu', 0.1],
#           'SQShift_768_2_8': [SQShift_768_2_8, 768, [(224, 2), (224, 4)], 8, 'relu', 0.1],
#           'SQShift_768_1_8': [SQShift_768_1_8, 768, [(224, 2)], 8, 'relu', 0.1],
#           'SQShift_128_2_8': [SQShift_128_2_8, 128, [(224, 2), (224, 4)], 8, 'relu', 0.1],
#           'SQShift_128_1_8': [SQShift_128_1_8, 128, [(224, 2)], 8, 'relu', 0.1],
#           'LWShift_768_2_8': [LWShift_768_2_8, 768, [(224, 2), (224, 4)], 8, 'relu', 0.1],
#           'LWShift_768_1_8': [LWShift_768_1_8, 768, [(224, 2)], 8, 'relu', 0.1],
#           'LWShift_128_2_8': [LWShift_128_2_8, 128, [(224, 2), (224, 4)], 8, 'relu', 0.1],
#           'LWShift_128_1_8': [LWShift_128_1_8, 128, [(224, 2)], 8, 'relu', 0.1]}

# for m in ['Shift', 'SQShift', 'LWShift']:
#     for dim in ['768_2', '768_1', '128_2', '128_1']:
#         model_name = f'{m}_{dim}_8'
#         target_model = models[model_name]

#         with open(save_path + f'{model_name}.txt', 'w') as f:
#             f.write(f'{m} model\n\n')
#             f.write(f'Patch Dimension: {target_model[1]}\n')
#             f.write(f'Patch Size: {target_model[2]}\n')
#             f.write(f'Number of Layers: {target_model[3]}\n')
#             f.write(f'Activation: {target_model[4]}\n')
#             f.write(f'Dropout: {target_model[5]}\n\n')
#             f.write(f'Model Parameters: {sum(p.numel() for p in target_model[0].parameters() if p.requires_grad)}\n\n')

#             for name, param in target_model[0].named_parameters():
#                 if param.requires_grad:
#                     f.write(f'{name}: {param.numel()}\n')

# Shift_128_1_7 = Shift.MultiscaleMixer(3, 128, 7, 0.1, patches=[(224, 2)], act='silu')
# Shift_128_1_6 = Shift.MultiscaleMixer(3, 128, 6, 0.1, patches=[(224, 2)], act='silu')
# Shift_128_1_5 = Shift.MultiscaleMixer(3, 128, 5, 0.1, patches=[(224, 2)], act='silu')
# Shift_128_1_4 = Shift.MultiscaleMixer(3, 128, 4, 0.1, patches=[(224, 2)], act='silu')

# models = {'Shift_128_1_7': [Shift_128_1_7, 128, [(224, 2)], 7, 'silu', 0.1],
#           'Shift_128_1_6': [Shift_128_1_6, 128, [(224, 2)], 6, 'silu', 0.1],
#           'Shift_128_1_5': [Shift_128_1_5, 128, [(224, 2)], 5, 'silu', 0.1],
#           'Shift_128_1_4': [Shift_128_1_4, 128, [(224, 2)], 4, 'silu', 0.1]}

# for dim in ['128_1_7', '128_1_6', '128_1_5', '128_1_4']:
#     model_name = f'Shift_{dim}'
#     target_model = models[model_name]

#     with open(save_path + f'{model_name}.txt', 'w') as f:
#         f.write(f'Shift model\n\n')
#         f.write(f'Patch Dimension: {target_model[1]}\n')
#         f.write(f'Patch Size: {target_model[2]}\n')
#         f.write(f'Number of Layers: {target_model[3]}\n')
#         f.write(f'Activation: {target_model[4]}\n')
#         f.write(f'Dropout: {target_model[5]}\n\n')
#         f.write(f'Model Parameters: {sum(p.numel() for p in target_model[0].parameters() if p.requires_grad)}\n\n')

#         for name, param in target_model[0].named_parameters():
#             if param.requires_grad:
#                 f.write(f'{name}: {param.numel()}\n')

MLP_Mixer_Small = MLPMixer(in_channels=3, dim=512, token_dim=256, channel_dim=2048, dropout=0.1, patch_size=16, image_size=224, depth=8, num_classes=6)

print(f'MLP Mixer Small model parameters: {sum(p.numel() for p in MLP_Mixer_Small.parameters() if p.requires_grad)}')