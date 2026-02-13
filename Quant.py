import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.ao.quantization as quantization
import torch.utils.data as data
import torchvision
import time
import os
import tempfile # 임시 파일 사용을 위해 import

from model.MSPS_Mixer_for_quant import MultiscaleMixer

def set_seed(seed=2024):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

def set_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using GPU:", torch.cuda.get_device_name(0))
    else:
        device = torch.device("cpu")
        print("Using CPU")

    return device

# 1. 모델 유틸리티 함수 (이전에 제공해 드린 함수)
def print_model_size(model, model_name="Model"):
    """모델의 파라미터 개수와 파일 크기(MB)를 계산합니다."""
    # ... (함수 내용 생략) ...
    # 모델 크기 및 파라미터 개수를 계산하는 실제 로직을 여기에 넣으세요.
    if not isinstance(model, torch.nn.Module):
        return 0.0
    
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "model.pt")
        torch.save(model.state_dict(), path)
        size_mb = os.path.getsize(path) / 1024**2
        
        # 파라미터 개수 계산
        num_params = sum(p.numel() for p in model.parameters())
        print(f"| {model_name:<15} | Size: {size_mb:.2f} MB | Params: {num_params:,} |")
        return size_mb

def time_model_inference(model, input_data, iterations=100):
    """모델의 추론 시간(평균)을 측정합니다."""
    # ... (함수 내용 생략) ...
    model.eval()
    times = []
    
    with torch.no_grad():
        # Warm-up
        for _ in range(10):
            model(input_data)
            
        # Measurement
        for _ in range(iterations):
            start_time = time.time()
            model(input_data)
            end_time = time.time()
            times.append(end_time - start_time)

    mean_time_ms = torch.tensor(times).mean().item() * 1000
    print(f"| Inference Time (avg): {mean_time_ms:.4f} ms |")
    return mean_time_ms


# 2. QAT 훈련 및 변환 실행 함수
def run_qat_pipeline(model_fp32, data_loader, criterion, optimizer, num_epochs, device):
    
    # 2-1. QConfig 설정 및 모델 준비
    
    # QAT QConfig 설정: 가중치(per-channel)와 활성화(per-tensor)에 Fake Quantization 적용
    # 'qnnpack'은 모바일 및 임베디드 장치에 최적화
    model_fp32.qconfig = quantization.get_default_qat_qconfig('qnnpack')
    print(f"QConfig 설정 완료: {model_fp32.qconfig}")

    # Fusion (선택 사항: Conv-BN-ReLU 등의 퓨전을 시도)
    # 현재 모델은 복잡한 사용자 정의 구조이므로, 명시적인 Fusion 대신 API에 맡깁니다.
    # fuse_modules() 대신 prepare_qat()가 내부적으로 퓨전을 시도합니다.

    # FakeQuantize 노드 삽입 및 Observer 활성화
    qat_model = quantization.prepare_qat(model_fp32, inplace=False)
    qat_model.train()
    print("모델 QAT 훈련 준비 완료 (Fake Quantization 노드 삽입)")

    # 2-2. QAT 훈련 루프
    print(f"\n--- QAT 훈련 시작 (Epochs: {num_epochs}) ---")
    for epoch in range(num_epochs):
        running_loss = 0.0
        start_time = time.time()
        
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = qat_model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        epoch_time = time.time() - start_time
        
        # 훈련 후반부 Observer 비활성화 (양자화 파라미터 확정)
        if epoch == num_epochs - 1: # 마지막 epoch에서 비활성화
             qat_model.apply(quantization.disable_observer)
             print("   [Observer Disabled]")
        
        print(f"Epoch [{epoch+1}/{num_epochs}] | Time: {epoch_time:.2f}s | Loss: {running_loss/len(data_loader):.4f}")

    # 2-3. 모델 양자화 변환 (Conversion)
    qat_model.eval()
    # FakeQuantize 노드를 실제 Quantized 모듈로 변환 (Int8)
    quantized_model = quantization.convert(qat_model, inplace=False)
    print("--- QAT 훈련 및 변환 완료: Int8 모델 획득 ---")
    
    return quantized_model.cpu()

# 4. ONNX 변환 및 저장 함수 추가
def export_quantized_model_to_onnx(model, dummy_input, onnx_path="quantized_multiscale_mixer.onnx"):
    """
    Int8 Quantized PyTorch 모델을 ONNX 형식으로 변환하여 저장합니다.
    """
    print("\n--- ONNX 변환 시작 ---")
    try:
        # Quantized 모델을 ONNX로 내보낼 때,
        # PyTorch는 Quantized Ops (예: nnq.Linear)를 ONNX Quantized Ops로 변환합니다.
        
        # 모델을 CPU로 이동 (일반적으로 ONNX 변환은 CPU에서 수행됨)
        model.cpu().eval()

        if dummy_input.device != torch.device('cpu'):
            dummy_input = dummy_input.cpu()
        
        # dynamic_axes: 가변적인 입력 크기를 지원하기 위해 (Batch Size) 설정
        torch.onnx.export(
            model,
            dummy_input, # 변환에 사용할 더미 입력 (B, C, H, W)
            onnx_path,
            export_params=True,
            opset_version=13, # Quantization 지원을 위해 Opset 13 이상 권장
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output']
        )
        
        # 파일 크기 확인
        size_mb = os.path.getsize(onnx_path) / 1024**2
        print(f"✅ ONNX 변환 성공! 파일 저장 경로: {onnx_path}")
        print(f"   ONNX File Size: {size_mb:.2f} MB")

    except Exception as e:
        print(f"❌ ONNX 변환 실패: {e}")
        # 참고: 복잡한 사용자 정의 모듈(예: channel_shift 내의 dequantize/quantize)은
        # ONNX 변환이 실패할 수 있으며, 이 경우 Torch Script로 우회하거나
        # 사용자 정의 ONNX 노드를 정의해야 할 수 있습니다.

# 3. Main 실행 함수
def main_qat_execution():
    # --- 하이퍼파라미터 및 설정 ---
    patch_dim = 128
    in_channels = 3
    output_classes = 6
    num_epochs = 30  # 실제 훈련 시 더 많은 epoch이 필요함
    B, H, W = 32, 224, 224 # Batch Size, Height, Width

    device = set_device()
    set_seed(2024)

    # 1. Float32 모델 초기화
    model_fp32 = MultiscaleMixer()
    model_fp32.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model_fp32.parameters(), lr=0.001)

    # 더미 데이터 로더 생성 (실제 데이터로 대체해야 함)
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((H, W)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    train_dataset = torchvision.datasets.ImageFolder(root='./data/IAA/train', transform=transform)
    train_loader = data.DataLoader(train_dataset, batch_size=B)


    # --- 2. QAT 파이프라인 실행 ---
    # 원본 모델의 가중치를 복사한 새 모델로 QAT를 진행합니다.
    model_to_qat = MultiscaleMixer()
    model_to_qat.load_state_dict(model_fp32.state_dict()) # 초기 가중치 복사
    model_to_qat.to(device)

    quantized_mixer = run_qat_pipeline(
        model_to_qat, train_loader, criterion, optimizer, num_epochs=num_epochs, device=device
    )

    # --- 3. 모델 크기 및 추론 속도 비교 ---
    print("\n" + "="*60)
    print("              [Float32 vs. Quantized Model 성능 비교]")
    print("="*60)

    dummy_input_data = torch.randn(1, 3, 224, 224).cpu()
    model_fp32.cpu()

    # 3-1. Float32 모델 성능 측정
    print("FP32 Model:")
    size_fp32 = print_model_size(model_fp32, "FP32 Model")
    time_fp32 = time_model_inference(model_fp32, dummy_input_data)
    
    # 3-2. Quantized 모델 성능 측정
    print("\nQuantized Model (Int8):")
    size_quantized = print_model_size(quantized_mixer, "Quantized Model")
    time_quantized = time_model_inference(quantized_mixer, dummy_input_data)
    
    print("\n" + "-"*60)
    # 계산은 실제 크기와 시간으로 수행
    size_reduction = (1 - size_quantized / size_fp32) * 100
    latency_improvement = (1 - time_quantized / time_fp32) * 100
    
    print(f"| File Size Reduction (vs. FP32): {size_reduction:.2f} % (기대: ~75%)")
    print(f"| Latency Improvement (vs. FP32): {latency_improvement:.2f} %")
    print("-"*60)

    print("\n--- ONNX 변환 시작 ---")
    export_quantized_model_to_onnx(
        quantized_mixer, 
        dummy_input=dummy_input_data, 
        onnx_path="quantized_multiscale_mixer.onnx"
    )

if __name__ == '__main__':
    main_qat_execution()