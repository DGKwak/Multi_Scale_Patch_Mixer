#!/bin/bash

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
uv run MSPM_test_main_with_SE.py data=IAA model.num_layers=8 experiment_name="MSPM_with_SE_IAA_test_8_layers" | tee ./logs/output_MSPM_with_SE_IAA_test_8_layers.log

# MSPM with SE CT - IAA
uv run MSPM_test_main_with_SE_CT.py data=IAA experiment_name="MSPM_with_SE_CT_IAA_test" | tee ./logs/output_MSPM_with_SE_CT_IAA_test.log

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