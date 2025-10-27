#!/bin/bash

# MSPM with Shift - Deferent shift test
# uv run main_new_shift.py experiment_name="MSPM_with_Shift_2_0_1" model.shift_l=[2,0,1] model.shift_r=[-1,0,-2]
# uv run main_new_shift.py experiment_name="MSPM_with_Shift_1_0_2" model.shift_l=[1,0,2] model.shift_r=[-2,0,-1]
# uv run main_new_shift.py experiment_name="MSPM_with_Shift_3_0_1" model.shift_l=[3,0,1] model.shift_r=[-1,0,-3]
# uv run main_new_shift.py experiment_name="MSPM_with_Shift_1_0_3" model.shift_l=[1,0,3] model.shift_r=[-3,0,-1]
# uv run main_new_shift.py experiment_name="MSPM_with_Shift_2_0_3" model.shift_l=[2,0,3] model.shift_r=[-3,0,-2]
# uv run main_new_shift.py experiment_name="MSPM_with_Shift_3_0_2" model.shift_l=[3,0,2] model.shift_r=[-2,0,-3]
# uv run main_new_shift.py experiment_name="MSPM_with_Shift_3_1_2" model.shift_l=[3,1,2] model.shift_r=[-2,-1,-3]

# MSPM with 1Shift and SE block
uv run main_1Shift_with_SE.py experiment_name="MSPM_1Shift_with_SE_CP_conv"

# CSM Freq patch
# uv run main.py experiment_name="CSM_Freq_patch_test" model.patches='[[16, 16], [4, 224]]'

# CSM 16x16 patch
# uv run main.py experiment_name="CSM_16x16_patch_test" model.patches='[[16, 16]]'

# CSM single patch 224x4
# uv run main.py experiment_name="CSM_single_patch_224_4_test" model.patches='[[224, 4]]'

# CSM single patch 224x2
# uv run main.py experiment_name="CSM_single_patch_224_2_test" model.patches='[[224, 2]]'

# CSM single patch 224x1
# uv run main.py experiment_name="CSM_single_patch_224_1_test" model.patches='[[224, 1]]'

# CSM patches 224x1, 224x2
# uv run main.py experiment_name="CSM_patches_224_1_2_test" model.patches='[[224, 1], [224, 2]]'

# CSM patches 224x1, 224x4
# uv run main.py experiment_name="CSM_patches_224_1_4_test" model.patches='[[224, 1], [224, 4]]'

# Comparison Model Test
# uv run Deit_main.py
# uv run Effi_main.py
# uv run Mobv1_main.py
# uv run Mobv2_main.py
# uv run MobViT_main.py

# New Shift
# uv run main_shiftatt.py experiment_name="New_Shift_Attention_test"

# New Residual CSM
# uv run main_New_res.py experiment_name="CSM_New_residual_test"
# uv run main_New_res.py experiment_name="CSM_New_Linear_128_4_test" model.patch_dim=128 model.num_layers=4

# CSM default test
# uv run MSPM_test_main_with_Shift.py experiment_name="CSM_default_test" | tee ./logs/output_CSM_default_test.log

# CSM default seed test
# uv run MSPM_test_main_with_Shift.py random_seed=1 experiment_name="CSM_default_seed1_test" | tee ./logs/output_CSM_default_seed1_test.log
# uv run MSPM_test_main_with_Shift.py random_seed=2 experiment_name="CSM_default_seed2_test" | tee ./logs/output_CSM_default_seed2_test.log
# uv run MSPM_test_main_with_Shift.py random_seed=50 experiment_name="CSM_default_seed50_test" | tee ./logs/output_CSM_default_seed50_test.log
# uv run MSPM_test_main_with_Shift.py random_seed=100 experiment_name="CSM_default_seed100_test" | tee ./logs/output_CSM_default_seed100_test.log
# uv run MSPM_test_main_with_Shift.py random_seed=52 experiment_name="CSM_default_seed52_test" | tee ./logs/output_CSM_default_seed52_test.log
# uv run MSPM_test_main_with_Shift.py random_seed=2024 experiment_name="CSM_default_seed2024_test" | tee ./logs/output_CSM_default_seed2024_test.log
# uv run MSPM_test_main_with_Shift.py random_seed=777 experiment_name="CSM_default_seed777_test" | tee ./logs/output_CSM_default_seed777_test.log
# uv run MSPM_test_main_with_Shift.py random_seed=999 experiment_name="CSM_default_seed999_test" | tee ./logs/output_CSM_default_seed999_test.log
# uv run MSPM_test_main_with_Shift.py random_seed=1024 experiment_name="CSM_default_seed1024_test" | tee ./logs/output_CSM_default_seed1024_test.log
# uv run MSPM_test_main_with_Shift.py random_seed=2048 experiment_name="CSM_default_seed2048_test" | tee ./logs/output_CSM_default_seed2048_test.log
# uv run MSPM_test_main_with_Shift.py random_seed=1000 experiment_name="CSM_default_seed777_test" | tee ./logs/output_CSM_default_seed777_test.log
# uv run MSPM_test_main_with_Shift.py random_seed=500 experiment_name="CSM_default_seed999_test" | tee ./logs/output_CSM_default_seed999_test.log
# uv run main.py --multirun random_state=2024,1,42,100,999 experiment_name='CSM_seed_2024_datasplit_712','CSM_seed_1_datasplit_712','CSM_seed_42_datasplit_712','CSM_seed_100_datasplit_712','CSM_seed_999_datasplit_712'

# CSM patch dim - 128 / layers 8, 4
# uv run main.py experiment_name="CSM_patch_dim_128_layers_8" model.patch_dim=128
# uv run main.py experiment_name="CSM_patch_dim_128_layers_4" model.num_layers=4 model.patch_dim=128

# CSM patch dim - 256 / layers 8, 4
# uv run main.py experiment_name="CSM_patch_dim_256_layers_8" model.patch_dim=256
# uv run main.py experiment_name="CSM_patch_dim_256_layers_4" model.num_layers=4 model.patch_dim=256

# CSM patch dim - 672 / layers 8, 4
# uv run main.py experiment_name="CSM_patch_dim_672_layers_8" model.patch_dim=672
# uv run main.py experiment_name="CSM_patch_dim_672_layers_4" model.num_layers=4 model.patch_dim=672

# CSM patch dim - 768 / layers 8, 4
# uv run main.py experiment_name="CSM_patch_dim_768_layers_8"
# uv run main.py experiment_name="CSM_patch_dim_768_layers_4" model.num_layers=4

# CSM with 1 shift block
# uv run MSPM_test_main_with_Shift.py experiment_name="CSM_with_1_Shift_test" | tee ./logs/output_CSM_with_1_Shift_test.log

# CSM with channel mixer F as Conv1d
# uv run MSPM_test_main_with_Shift.py experiment_name="CSM_with_Channel_Mixer_F_as_Conv1d_test" | tee ./logs/output_CSM_with_Channel_Mixer_F_as_Conv1d_test.log

# MSPM with Shift block - IAA layers 8
# uv run MSPM_test_main_with_Shift.py data=IAA model.num_layers=8 experiment_name="MSPM_with_Shift_IAA_test_8_layers" | tee ./logs/output_MSPM_with_Shift_IAA_test_8_layers.log

# MSPM with Shift block - IAA layers 8 single patch (224,2)
# uv run MSPM_test_main_with_Shift.py data=IAA model.num_layers=8 model.patches='[[224, 2]]' experiment_name="MSPM_with_Shift_IAA_test_8_layers_single_patch" | tee ./logs/output_MSPM_with_Shift_IAA_test_8_layers_single_patch.log

# MSPM with Shift block - IAA layers 8 patch_dim 128
# uv run MSPM_test_main_with_Shift.py data=IAA model.num_layers=8 model.patch_dim=128 experiment_name="MSPM_with_Shift_IAA_test_8_layers_patch_dim_128" | tee ./logs/output_MSPM_with_Shift_IAA_test_8_layers_patch_dim_128.log

# MSPM with Shift block - IAA layers 8 patch_dim 128 single patch (224,2)
# uv run MSPM_test_main_with_Shift.py data=IAA model.num_layers=8 model.patch_dim=128 model.patches='[[224, 2]]' experiment_name="MSPM_with_Shift_IAA_test_8_layers_patch_dim_128_single_patch" | tee ./logs/output_MSPM_with_Shift_IAA_test_8_layers_patch_dim_128_single_patch.log

# MSPM with Shift block - IAA layers 8 patch_dim 128 single patch (224,2) with silu activation
# uv run MSPM_test_main_with_Shift.py data=IAA model.num_layers=8 model.patch_dim=128 model.patches='[[224, 2]]' model.activation='silu' experiment_name="MSPM_with_Shift_IAA_test_8_layers_patch_dim_128_single_patch_silu" | tee ./logs/output_MSPM_with_Shift_IAA_test_8_layers_patch_dim_128_single_patch_silu.log

# MSPM with Shift block - IAA layers 7 patch_dim 128 single patch (224,2) with silu activation
# uv run MSPM_test_main_with_Shift.py data=IAA model.num_layers=7 model.patch_dim=128 model.patches='[[224, 2]]' model.activation='silu' experiment_name="MSPM_with_Shift_IAA_test_7_layers_patch_dim_128_single_patch_silu" | tee ./logs/output_MSPM_with_Shift_IAA_test_7_layers_patch_dim_128_single_patch_silu.log

# MSPM with Shift block - IAA layers 6 patch_dim 128 single patch (224,2) with silu activation
# uv run MSPM_test_main_with_Shift.py data=IAA model.num_layers=6 model.patch_dim=128 model.patches='[[224, 2]]' model.activation='silu' experiment_name="MSPM_with_Shift_IAA_test_6_layers_patch_dim_128_single_patch_silu" | tee ./logs/output_MSPM_with_Shift_IAA_test_6_layers_patch_dim_128_single_patch_silu.log

# MSPM with Shift block - IAA layers 5 patch_dim 128 single patch (224,2) with silu activation
# uv run MSPM_test_main_with_Shift.py data=IAA model.num_layers=5 model.patch_dim=128 model.patches='[[224, 2]]' model.activation='silu' experiment_name="MSPM_with_Shift_IAA_test_5_layers_patch_dim_128_single_patch_silu" | tee ./logs/output_MSPM_with_Shift_IAA_test_5_layers_patch_dim_128_single_patch_silu.log

# MSPM with Shift block - IAA layers 4 patch_dim 128 single patch (224,2) with silu activation
# uv run MSPM_test_main_with_Shift.py data=IAA model.num_layers=4 model.patch_dim=128 model.patches='[[224, 2]]' model.activation='silu' experiment_name="MSPM_with_Shift_IAA_test_4_layers_patch_dim_128_single_patch_silu" | tee ./logs/output_MSPM_with_Shift_IAA_test_4_layers_patch_dim_128_single_patch_silu.log

# Lightweight MSPM with Shift block - IAA layers 8 patch_dim 128
# uv run Lightweight_MSPM_test_main_with_Shift.py data=IAA model.num_layers=8 model.patch_dim=128 experiment_name="Lightweight_MSPM_with_Shift_IAA_test_8_layers_patch_dim_128" | tee ./logs/output_Lightweight_MSPM_with_Shift_IAA_test_8_layers_patch_dim_128.log

# Lightweight MSPM with Shift block - IAA layers 8 patch_dim 128 single patch (224,2)
# uv run Lightweight_MSPM_test_main_with_Shift.py data=IAA model.num_layers=8 model.patch_dim=128 model.patches='[[224, 2]]' experiment_name="Lightweight_MSPM_with_Shift_IAA_test_8_layers_patch_dim_128_single_patch" | tee ./logs/output_Lightweight_MSPM_with_Shift_IAA_test_8_layers_patch_dim_128_single_patch.log

# MSPM with Shift block Squeezed - IAA layers 8 patch_dim 128 silu activation
# uv run MSPM_test_main_with_Shift_Squeezed.py data=IAA model.num_layers=8 model.activation='silu' model.patch_dim=128 experiment_name="MSPM_with_Shift_Squeezed_IAA_test_8_layers_patch_dim_128" | tee ./logs/output_MSPM_with_Shift_Squeezed_IAA_test_8_layers_patch_dim_128_silu.log

# MSPM with Shift block Squeezed - IAA layers 8 patch_dim 128 single patch (224,2) silu activation
# uv run MSPM_test_main_with_Shift_Squeezed.py data=IAA model.num_layers=8 model.activation='silu' model.patch_dim=128 model.patches='[[224, 2]]' experiment_name="MSPM_with_Shift_Squeezed_IAA_test_8_layers_patch_dim_128_single_patch" | tee ./logs/output_MSPM_with_Shift_Squeezed_IAA_test_8_layers_patch_dim_128_single_patch_silu.log

# MSPM with Shift block Squeezed - IAA layers 8 patch_dim 128 single patch (224,2) gelu activation
# uv run MSPM_test_main_with_Shift_Squeezed.py data=IAA model.num_layers=8 model.activation='gelu' model.patch_dim=128 model.patches='[[224, 2]]' experiment_name="MSPM_with_Shift_Squeezed_IAA_test_8_layers_patch_dim_128_single_patch_gelu" | tee ./logs/output_MSPM_with_Shift_Squeezed_IAA_test_8_layers_patch_dim_128_single_patch_gelu.log

# MSPM with Shift block Squeezed - IAA layers 8 patch_dim 128 single patch (224,4) gelu activation
# uv run MSPM_test_main_with_Shift_Squeezed.py data=IAA model.num_layers=8 model.activation='gelu' model.patch_dim=128 model.patches='[[224, 4]]' experiment_name="MSPM_with_Shift_Squeezed_IAA_test_8_layers_patch_dim_128_single_patch4_gelu" | tee ./logs/output_MSPM_with_Shift_Squeezed_IAA_test_8_layers_patch_dim_128_single_patch4_gelu.log

# patch_dim : 768
# uv run MSPM_test_main_with_test.py experiment_name="MSPM_patch_dim_768_test" | tee ./logs/output_MSPM_patch_dim_768_test.log

# patch_dim : 768 with pyramid structure
# uv run MSPM_test_main_with_test.py experiment_name="MSPM_patch_dim_768_with_pyramid_test" | tee ./logs/output_MSPM_patch_dim_768_with_pyramid_test.log

# FTMixer with cosine similarity loss
# uv run FTMixer_test_main_with_cos_sim.py experiment_name="FTMixer_with_cosine_similarity_loss_test" | tee ./logs/output_FTMixer_with_cosine_similarity_loss_test.log

# FTMixer with cosine similarity loss - layer 8
# uv run FTMixer_test_main_with_cos_sim.py model.num_layers=8 experiment_name="FTMixer_with_cosine_similarity_loss_8_layers_test" | tee ./logs/output_FTMixer_with_cosine_similarity_loss_8_layers_test.log

# FTMixer with cosine similarity loss - IAA
# uv run FTMixer_test_main_with_cos_sim.py data=IAA experiment_name="FTMixer_with_cosine_similarity_loss_IAA_test" | tee ./logs/output_FTMixer_with_cosine_similarity_loss_IAA_test.log

# MSPM with SE - IAA
# uv run MSPM_test_main_with_SE.py data=IAA experiment_name="MSPM_with_SE_IAA_test" | tee ./logs/output_MSPM_with_SE_IAA_test.log

# MSPM with SE - IAA and 8 layers
# uv run MSPM_test_main_with_SE.py data=IAA model.num_layers=8 experiment_name="MSPM_with_SE_IAA_test_8_layers" | tee ./logs/output_MSPM_with_SE_IAA_test_8_layers.log

# MSPM with SE CT - IAA
# uv run MSPM_test_main_with_SE_CT.py data=IAA experiment_name="MSPM_with_SE_CT_IAA_test" | tee ./logs/output_MSPM_with_SE_CT_IAA_test.log

# MSPM with 8 layers
# uv run MSPM_test_main_with_test.py model.num_layers=8 experiment_name="MSPM_8_layers_test" | tee ./logs/output_MSPM_8_layers_test.log

# MSPM with (16, 16) patch
# uv run MSPM_test_main_with_test.py model.patches='[[16, 16]]' experiment_name="MSPM_16_16_patch_test" | tee ./logs/output_MSPM_16_16_patch_test.log

# Augmentation test
# uv run MSPM_test_main_with_customdataset.py experiment_name="Augmentation_test" | tee ./logs/output_Augmentation_test.log
# uv run MSPM_test_main_with_customdataset.py model.patches='[[224, 1], [224, 2], [224, 4]]' experiment_name="Augmentation_3_Patches_test" | tee ./logs/output_Augmentation_3patches_test.log

# STFT Original MLP-Mixer Test
# uv run Original_MLP_Mixer_test.py | tee ./logs/output_Original_MLP_Mixer_test.log

# STFT MSPM Test
# uv run MSPM_test_main_with_test.py | tee ./logs/output_MSPM_test.log

# IAA Test
# uv run Original_MLP_Mixer_test.py data=IAA | tee ./logs/output_Original_MLP_Mixer_IAA_test.log
# uv run MSPM_test_main_with_test.py data=IAA | tee ./logs/output_MSPM_IAA_test.log

# IAA Test
# uv run MSPM_test_main_ori.py data.data_dir='/home/eslab/Vscode/test_model/data/IAA' experiment_name="MSPM_IAA_test" | tee ./logs/output_MSPM_IAA_test.log

# STFT + Flipped STFT
# uv run MSPM_test_main_ori.py data.data_dir='/home/eslab/Vscode/test_model/data/STFT_and_Flip' experiment_name="MSPM_STFT_and_flipped_test" | tee ./logs/output_MSPM_STFT_and_Flip_test.log

# STFT + Flipped STFT + Shifted STFT
# uv run MSPM_test_main_ori.py data.data_dir='/home/eslab/Vscode/test_model/data/STFT_Flip_Shift' experiment_name="MSPM_STFT_and_flipped_test" | tee ./logs/output_MSPM_STFT_Flip_Shift_test.log

# IAA + Flipped IAA + Shifted IAA
# uv run MSPM_test_main_ori.py data.data_dir='/home/eslab/Vscode/test_model/data/Merged_IAA' experiment_name="MSPM_IAA_and_flipped_test" | tee ./logs/output_MSPM_IAA_Flip_Shift_test.log

# New Dataset test
# uv run MSPM_test_main_ori.py data.data_dir='/home/eslab/Vscode/test_model/data/Merged_data' experiment_name="MSPM_New_Data_test" | tee ./logs/output_MSPM_New_Data_test.log

# New Dataset with 3 patches
# uv run MSPM_test_main_ori.py data.data_dir='/home/eslab/Vscode/test_model/data/Merged_data' model.patches='[[224, 1], [224, 2], [224, 4]]' experiment_name="MSPM_New_Data_3_Patches" | tee ./logs/output_MSPM_New_Data_3_Patches.log

# New Dataset original patch
# uv run MSPM_test_main_ori.py data.data_dir='/home/eslab/Vscode/test_model/data/Merged_data' model.patches='[[16, 16]]' experiment_name="MSPM_New_Data_Original_Patch" | tee ./logs/output_MSPM_New_Data_Original_Patch.log

# New Dataset linear patchify test 
# uv run MSPM_Linear_Patchify.py data.data_dir='/home/eslab/Vscode/test_model/data/Merged_data' experiment_name="MSPM_New_Data_Linear_Patchify_test" | tee ./logs/output_MSPM_New_Data_Linear_Patchify_test.log

# patches = [224, 1], [224, 2], [224, 4] 3 patches
# uv run MSPM_test_main_ori.py model.patches='[[224, 1], [224, 2], [224, 4]]' experiment_name="MSPM_3_Patches" | tee ./logs/output_MSPM_3_Patches.log

# linear patchify test
# uv run MSPM_Linear_Patchify.py experiment_name="MSPM_Linear_Patchify_test" | tee ./logs/output_MSPM_Linear_Patchify_test.log

# linear patchify test with 3 patches
# uv run MSPM_Linear_Patchify_test.py model.patches='[[224, 1], [224, 2], [224, 4]]' experiment_name="MSPM_Linear_Patchify_test" | tee ./logs/output_MSPM_Linear_Patchify_test.log

# reference code test
# HireMLP
# uv run test_reference_codes.py | tee ./logs/output_test_reference_codes_HireMLP.log

# WaveMLP
# uv run test_reference_codes.py model_name="WaveMLP" | tee ./logs/output_test_reference_codes_WaveMLP.log