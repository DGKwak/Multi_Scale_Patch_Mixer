import model.Multi_Scale_Patch_Mixer_with_Shift as Shift
import model.Multi_Scale_Patch_Mixer_with_Shift_Squeezed_hidden as SQShift
import model.Lightweight_Multi_Scale_Patch_Mixer_with_Shift as LWShift
from Reference_code.MLP_Mixer import MLPMixer
from model.CSM_New_residual import MultiscaleMixer
from model.CSM_revision05 import MultiscaleMixer as CSM_Rev05
from model.CSM_low_rank import MultiscaleMixer as CSM_Low_Rank
from model.Multi_Scale_Shift_Mixer_GLU import MultiscaleMixer as MSSM_GLU

from Reference_code.Comparison_model.DeiT import model as Deit
from Reference_code.Comparison_model.EfficientNet import model as EfficientNet
from Reference_code.Comparison_model.MobileNetv1 import model as MobileNetv1
from Reference_code.Comparison_model.MobileNetv2 import model as MobileNetv2
from Reference_code.Comparison_model.MobileViT import model as MobileViT
from Reference_code.Comparison_model.ShuffleNet import model as ShuffleNet

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

# save_path = './results/model_param/'

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

# MLP_Mixer_Small = MLPMixer(in_channels=3, dim=512, token_dim=256, channel_dim=2048, dropout=0.1, patch_size=16, image_size=224, depth=8, num_classes=6)
# print(f'MLP Mixer Small model parameters: {sum(p.numel() for p in MLP_Mixer_Small.parameters() if p.requires_grad)}')

# model = Deit.deit_tiny_patch16_224(pretrained=False, num_classes=6)
# print('DeiT Parameter Count:', sum(p.numel() for p in model.parameters() if p.requires_grad))

# model = EfficientNet.EfficientNet.from_name('efficientnet-b0', num_classes=6)
# print('EfficientNet Parameter Count:', sum(p.numel() for p in model.parameters() if p.requires_grad))

# model = MobileNetv1.MobileNetV1(ch_in=3, n_classes=6)
# print('MobileNetV1 Parameter Count:', sum(p.numel() for p in model.parameters() if p.requires_grad))

# model = MobileNetv2.MobileNetV2(ch_in=3, n_classes=6)
# print('MobileNetV2 Parameter Count:', sum(p.numel() for p in model.parameters() if p.requires_grad))

# model = MobileViT.mobilevit_xxs_for_test()
# print('MobileViT Parameter Count:', sum(p.numel() for p in model.parameters() if p.requires_grad))

# model = ShuffleNet.ShuffleNet(num_classes=6)
# print('ShuffleNet Parameter Count:', sum(p.numel() for p in model.parameters() if p.requires_grad))

# model = MultiscaleMixer(3, 768, 8, 0.1, act='relu')
# print('CSM New Residual Parameter Count:', sum(p.numel() for p in model.parameters() if p.requires_grad))

# model = CSM_Rev05(3, 786, 8, 0.1, act='relu')
# print('CSM Revision05 Parameter Count:', sum(p.numel() for p in model.parameters() if p.requires_grad))

# model = Shift.MultiscaleMixer(3, 128, 4, 0.1, act='relu')
# print('Original Shift 128_4 Parameter Count:', sum(p.numel() for p in model.parameters() if p.requires_grad))

# model = Shift.MultiscaleMixer(3, 128, 8, 0.1, act='relu')
# print('Original Shift 128_8 Parameter Count:', sum(p.numel() for p in model.parameters() if p.requires_grad))

# model = Shift.MultiscaleMixer(3, 256, 4, 0.1, act='relu')
# print('Original Shift 256_4 Parameter Count:', sum(p.numel() for p in model.parameters() if p.requires_grad))

# model = Shift.MultiscaleMixer(3, 256, 8, 0.1, act='relu')
# print('Original Shift 256_8 Parameter Count:', sum(p.numel() for p in model.parameters() if p.requires_grad))

# model = Shift.MultiscaleMixer(3, 768, 4, 0.1, act='relu')
# print('Original Shift 768_4 Parameter Count:', sum(p.numel() for p in model.parameters() if p.requires_grad))

# model = Shift.MultiscaleMixer(3, 768, 8, 0.1, act='relu')
# print('Original Shift 768_8 Parameter Count:', sum(p.numel() for p in model.parameters() if p.requires_grad))

# model = CSM_Low_Rank(3, 768, 8, 0.1, act='relu')
# print('CSM Low Rank Parameter Count:', sum(p.numel() for p in model.parameters() if p.requires_grad))

model = MSSM_GLU(3, 128, 0.1, [2, 2], shift=[3, -2, 2, -3], shift_size=4, act='relu')
print('MSSM GLU Parameter Count:', sum(p.numel() for p in model.parameters() if p.requires_grad))