# Multi_Scale_Patch_Mixer

### How to Use
* 파이썬 uv를 사용하여 환경 구축
    * pip install uv로 uv 설치
    * git clone 이후 uv sync로 가상환경 구축
    * uv run MSPM_test_main.ori.py로 실행

### 파일 구조
* 모델 파일 : ./Multi_Scale_Patch_Mixer_ori.py
* 테스트 실행 파일 : ./MSPM_test_main_ori.py
* loss 함수 파일 : ./loss/loss_func.py
* 테스트 설정 : ./config (hydra 기반 yaml 파일 사용)
    * config.yaml
        * 테스트 설정값
        * 시용시 csv_path와 best_model_path 수정 필요
    * model/default.yaml
        * 모델 파라미터
    * loss/loss.yaml
        * loss 함수 파라미터
    * data/transform.yaml
        * 이미지 transform 설정
    * data/UoG.yaml
        * 데이터 설정
        * 사용시 data_dir 수정 필요