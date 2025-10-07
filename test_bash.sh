#!/bin/bash

# patches = [224, 1], [224, 2], [224, 4] 3 patches
# uv run MSPM_test_main_ori.py model.patches='[[224, 1], [224, 2], [224, 4]]' experiment_name="MSPM_3_Patches" | tee ./logs/output_MSPM_3_Patches.log

# linear patchify test
uv run MSPM_Linear_Patchify.py experiment_name="MSPM_Linear_Patchify_test" | tee ./logs/output_MSPM_Linear_Patchify_test.log

# linear patchify test with 3 patches
uv run MSPM_Linear_Patchify_test.py model.patches='[[224, 1], [224, 2], [224, 4]]' experiment_name="MSPM_Linear_Patchify_test" | tee ./logs/output_MSPM_Linear_Patchify_test.log