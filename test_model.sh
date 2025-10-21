#! /bin/bash

# test data - IAA Sobel_3 KFold
uv run CSM_main_with_Shift_newData_kfold.py test_model_name="horizontal_sliding" experiment_name="CSM_test_model_kfold_horizontal_sliding" | tee ./logs/test_model_kfold_horizontal_sliding.log

uv run CSM_main_with_Shift_newData_kfold.py test_model_name="test_model_768_8_2_01" experiment_name="CSM_test_model_kfold_768_8_2_01" | tee ./logs/test_model_kfold_768_8_2_01.log
uv run CSM_main_with_Shift_newData_kfold.py test_model_name="test_model_768_8_2_02" experiment_name="CSM_test_model_kfold_768_8_2_02" | tee ./logs/test_model_kfold_768_8_2_02.log
uv run CSM_main_with_Shift_newData_kfold.py test_model_name="test_model_768_4_2_01" experiment_name="CSM_test_model_kfold_768_4_2_01" | tee ./logs/test_model_kfold_768_4_2_01.log
uv run CSM_main_with_Shift_newData_kfold.py test_model_name="test_model_768_4_2_02" experiment_name="CSM_test_model_kfold_768_4_2_02" | tee ./logs/test_model_kfold_768_4_2_02.log
uv run CSM_main_with_Shift_newData_kfold.py test_model_name="test_model_128_8_2_01" experiment_name="CSM_test_model_kfold_128_8_2_01" | tee ./logs/test_model_kfold_128_8_2_01.log
uv run CSM_main_with_Shift_newData_kfold.py test_model_name="test_model_128_8_2_02" experiment_name="CSM_test_model_kfold_128_8_2_02" | tee ./logs/test_model_kfold_128_8_2_02.log
uv run CSM_main_with_Shift_newData_kfold.py test_model_name="test_model_128_4_2_01" experiment_name="CSM_test_model_kfold_128_4_2_01" | tee ./logs/test_model_kfold_128_4_2_01.log
uv run CSM_main_with_Shift_newData_kfold.py test_model_name="test_model_128_4_2_02" experiment_name="CSM_test_model_kfold_128_4_2_02" | tee ./logs/test_model_kfold_128_4_2_02.log

uv run CSM_main_with_Shift_newData_kfold.py data=IAA_DWT test_model_name="test_model_IAA_DWT_768_8_2_01" experiment_name="CSM_test_model_kfold_IAA_DWT_768_8_2_01" | tee ./logs/test_model_kfold_IAA_DWT_768_8_2_01.log
uv run CSM_main_with_Shift_newData_kfold.py data=IAA_DWT test_model_name="test_model_IAA_DWT_768_8_2_02" experiment_name="CSM_test_model_kfold_IAA_DWT_768_8_2_02" | tee ./logs/test_model_kfold_IAA_DWT_768_8_2_02.log
uv run CSM_main_with_Shift_newData_kfold.py data=IAA_DWT test_model_name="test_model_IAA_DWT_768_4_2_01" experiment_name="CSM_test_model_kfold_IAA_DWT_768_4_2_01" | tee ./logs/test_model_kfold_IAA_DWT_768_4_2_01.log
uv run CSM_main_with_Shift_newData_kfold.py data=IAA_DWT test_model_name="test_model_IAA_DWT_768_4_2_02" experiment_name="CSM_test_model_kfold_IAA_DWT_768_4_2_02" | tee ./logs/test_model_kfold_IAA_DWT_768_4_2_02.log
uv run CSM_main_with_Shift_newData_kfold.py data=IAA_DWT test_model_name="test_model_IAA_DWT_128_8_2_01" experiment_name="CSM_test_model_kfold_IAA_DWT_128_8_2_01" | tee ./logs/test_model_kfold_IAA_DWT_128_8_2_01.log
uv run CSM_main_with_Shift_newData_kfold.py data=IAA_DWT test_model_name="test_model_IAA_DWT_128_8_2_02" experiment_name="CSM_test_model_kfold_IAA_DWT_128_8_2_02" | tee ./logs/test_model_kfold_IAA_DWT_128_8_2_02.log
uv run CSM_main_with_Shift_newData_kfold.py data=IAA_DWT test_model_name="test_model_IAA_DWT_128_4_2_01" experiment_name="CSM_test_model_kfold_IAA_DWT_128_4_2_01" | tee ./logs/test_model_kfold_IAA_DWT_128_4_2_01.log
uv run CSM_main_with_Shift_newData_kfold.py data=IAA_DWT test_model_name="test_model_IAA_DWT_128_4_2_02" experiment_name="CSM_test_model_kfold_IAA_DWT_128_4_2_02" | tee ./logs/test_model_kfold_IAA_DWT_128_4_2_02.log