import torch
import torch.nn as nn
import os
import sys
from model.MSPS_Mixer import MultiscaleMixer

# 가정: MultiscaleMixer 클래스는 import 가능해야 합니다.
# from model.MSPS_Mixer import MultiscaleMixer 

# ------------------------------------------------------------------
# 1. 유틸리티 함수 (기존 코드에서 가져옴)
# ------------------------------------------------------------------

def count_total_parameters(model):
    """모델이 메모리에 로드한 전체 파라미터 수 (모든 weight와 bias)를 계산합니다."""
    total_params = sum(p.numel() for p in model.parameters())
    return total_params

def count_active_parameters(model):
    """프루닝을 통해 0이 되지 않은, 실제로 유효한 파라미터 수를 계산합니다."""
    active_params = 0
    
    for module in model.modules():
        # nn.Linear, nn.Conv2d 등 가중치를 가진 모듈만 필터링
        if isinstance(module, (nn.Conv2d, nn.Linear, nn.Conv1d)):
            
            # weight 확인: prune.remove() 후 저장된 파일이므로, 
            # module.weight에는 이미 0이 적용되어 있습니다. 0이 아닌 원소의 개수를 셉니다.
            if hasattr(module, 'weight') and module.weight is not None:
                # torch.count_nonzero는 0이 아닌 원소의 개수를 효율적으로 계산합니다.
                active_params += torch.count_nonzero(module.weight).item()

            # bias 확인: 편향(Bias)은 프루닝 대상에서 제외했으므로 모두 유효하다고 가정
            if hasattr(module, 'bias') and module.bias is not None:
                active_params += module.bias.numel()
                
    return active_params

# ------------------------------------------------------------------
# 2. 메인 실행 블록
# ------------------------------------------------------------------

if __name__ == "__main__":
    # 📌 설정 변경: 로드할 파일 경로와 모델 초기화 인자를 정확히 입력하세요.
    PRUNED_MODEL_PATH = 'MSPS_Pruned.pth'
    
    # 🚨🚨 MultiscaleMixer 초기화 인자는 사용자의 실제 모델 구조에 맞춰야 합니다. 🚨🚨
    # 예시 인자: (실제 모델 초기화 시 사용한 인자로 대체해야 함)
    # model = MultiscaleMixer(
    #     in_channels=3, 
    #     patch_dim=768, 
    #     num_classes=10, 
    #     num_layers=12,
    #     ...)
    
    # 예시를 위해 단순화된 모델 초기화 코드를 사용했습니다.
    # 만약 MultiscaleMixer가 인자가 필요하다면 위에 주석 처리된 부분처럼 인자를 넣어주세요.
    try:
        # MultiscaleMixer 모델 구조를 다시 생성
        # (주의: MultiscaleMixer()가 인자 없이 호출 가능하다고 가정했습니다.)
        model = MultiscaleMixer() 
        device = torch.device("cpu") # 파라미터 확인은 CPU에서 충분합니다.
        model.to(device)

        # 1. 프루닝된 가중치 로드
        print(f"가중치 파일 로드 중: {PRUNED_MODEL_PATH}")
        # map_location을 'cpu'로 지정하여 메모리 사용량을 줄일 수 있습니다.
        state_dict = torch.load(PRUNED_MODEL_PATH, map_location=device)
        model.load_state_dict(state_dict)
        print("가중치 로드 완료.")

        # 2. 파라미터 수 확인
        total_params = count_total_parameters(model)
        active_params = count_active_parameters(model)

        # 3. 결과 출력
        print("-" * 50)
        print("✨ 프루닝 모델 파라미터 분석 결과 ✨")
        print(f"1. 논리적 전체 파라미터 수 (Total Parameters): {total_params:,} 개")
        print(f"2. 유효 파라미터 수 (Active/Non-zero Parameters): {active_params:,} 개")
        
        # 0이 아닌 파라미터가 전체에서 차지하는 비율
        active_ratio = active_params / total_params
        
        print(f"3. 제거된 파라미터 비율 (희소도): {(1 - active_ratio) * 100:.2f}%")
        print(f"4. 남아있는 파라미터 비율: {active_ratio * 100:.2f}%")
        print("-" * 50)

    except FileNotFoundError:
        print(f"오류: '{PRUNED_MODEL_PATH}' 파일을 찾을 수 없습니다. 경로를 확인해 주세요.")
    except Exception as e:
        print(f"오류: 모델 로드 또는 파라미터 계산 중 문제가 발생했습니다: {e}")