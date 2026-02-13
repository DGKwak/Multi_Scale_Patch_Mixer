import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.utils.prune as prune
import torchvision
import math

from model.MSPS_Mixer import MultiscaleMixer

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def measure_global_sparsity(model):
    """모델의 전체 가중치 중 0의 비율(희소도)을 계산합니다."""
    num_zeros = 0
    num_elements = 0
    
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear, nn.Conv1d)):
            # 'weight'는 프루닝 마스크가 적용된 최종 값을 반환합니다.
            if hasattr(module, 'weight') and module.weight is not None:
                num_zeros += torch.sum(module.weight == 0).item()
                num_elements += module.weight.nelement()
            
    if num_elements == 0:
        return 0.0
        
    sparsity = num_zeros / num_elements
    return sparsity

def get_linear_pruning_targets(model):
    """프루닝을 적용할 nn.Linear 레이어의 (모듈, 'weight') 튜플 리스트를 반환합니다."""
    parameters_to_prune = []
    
    for module in model.modules():
        if isinstance(module, nn.Linear):
            # nn.Linear의 가중치(weight)에 프루닝을 적용합니다.
            parameters_to_prune.append((module, 'weight'))
            
    return parameters_to_prune

def finalize_pruned_model(parameters_to_prune):
    """프루닝 마스크를 제거하고 최종 0으로 고정된 모델을 만듭니다."""
    print("\n[INFO] Pruning 마스크를 제거하고 모델을 확정합니다...")
    for module, name in parameters_to_prune:
        # 프루닝이 적용된 모듈만 제거
        if prune.is_pruned(module):
            prune.remove(module, name)
            
    print("[SUCCESS] Pruning finalized. 모델이 추론 준비 완료 상태가 되었습니다.")

def train_one_epoch(model, dataloader, optimizer, criterion, device):
    """데이터로더를 사용하여 모델을 1 에포크 학습시키는 함수입니다."""
    model.train()
    running_loss = 0.0
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        
        outputs, z = model(inputs) # MultiscaleMixer의 출력 구조에 맞춤
        loss = criterion(outputs, labels)
        aux_loss = 0

        for out in z:
            if out.ndim > 2:
                out = torch.mean(out, dim=2, keepdim=False)

            aux_loss += criterion(out, labels)
        
        loss += 0.3 * aux_loss

        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    # print(f"  [Train] Loss: {running_loss / len(dataloader):.4f}")
    return running_loss

# 🚨 사용자 정의 필요 🚨
def evaluate_model(model, dataloader, criterion, device):
    """데이터로더를 사용하여 모델의 정확도와 손실을 평가하는 함수입니다."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs, _ = model(inputs)
            
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    accuracy = 100 * correct / total
    print(f"  [Eval] Loss: {total_loss / len(dataloader):.4f}, Acc: {accuracy:.2f}%")
    return accuracy

def run_iterative_pruning(model, train_loader, test_loader, optimizer, criterion, device):
    
    # ------------------------------------------------------------------
    # 💡 프루닝 하이퍼파라미터 설정
    # ------------------------------------------------------------------
    NUM_ITERATIONS = 5      # 프루닝을 총 몇 번 반복할지
    PRUNE_EPOCHS = 10       # 프루닝 한 번 적용 후 몇 에포크 파인튜닝할지
    TOTAL_PRUNE_AMOUNT = 0.70 # 최종 목표 희소도 (전체 Linear 가중치의 70% 제거 목표)

    # 매번 반복마다 제거할 비율 계산 (남은 가중치 대비)
    # (1 - r) = (1 - TOTAL_PRUNE_AMOUNT) ^ (1 / NUM_ITERATIONS) 공식을 사용
    prune_ratio_per_iteration = 1 - math.pow(1 - TOTAL_PRUNE_AMOUNT, 1 / NUM_ITERATIONS)
    
    print(f"[Config] 목표 희소도: {TOTAL_PRUNE_AMOUNT*100:.1f}%, 반복 횟수: {NUM_ITERATIONS}회")
    print(f"[Config] 매회 제거 비율 (남은 가중치 대비): {prune_ratio_per_iteration*100:.2f}%")
    print("-" * 50)
    
    # 1. 프루닝 대상 수집
    parameters_to_prune = get_linear_pruning_targets(model)

    # 초기 성능 및 희소도 측정
    initial_acc = evaluate_model(model, test_loader, criterion, device)
    initial_sparsity = measure_global_sparsity(model)
    print(f"[Initial] Sparsity: {initial_sparsity:.4f}, Accuracy: {initial_acc:.2f}%")
    print("-" * 50)

    # 2. 반복 루프 시작
    for i in range(NUM_ITERATIONS):
        print(f"\n======== ITERATION {i+1}/{NUM_ITERATIONS} ========")

        # A. 프루닝 적용 (Global Unstructured Pruning)
        prune.global_unstructured(
            parameters_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=prune_ratio_per_iteration,
        )
        
        current_sparsity = measure_global_sparsity(model)
        print(f"[Pruning] Sparsity after Pruning: {current_sparsity:.4f}")

        # B. 파인튜닝 (재학습)
        print(f"[Fine-tuning] Starting Fine-tuning for {PRUNE_EPOCHS} epochs...")
        for epoch in range(PRUNE_EPOCHS):
            # 모델을 프루닝 마스크가 적용된 상태로 학습
            train_one_epoch(model, train_loader, optimizer, criterion, device)
            # 매 에포크마다 성능 평가 (선택 사항)
            acc = evaluate_model(model, test_loader, criterion, device)
            print(f"  [Epoch {epoch+1}/{PRUNE_EPOCHS}] Accuracy: {acc:.2f}%")

        # C. 반복 후 최종 성능 확인
        final_acc = evaluate_model(model, test_loader, criterion, device)
        print(f"[RESULT] Iteration {i+1} Final Accuracy: {final_acc:.2f}%")

    # 3. 프루닝 확정 및 모델 정리
    finalize_pruned_model(parameters_to_prune)
    final_sparsity = measure_global_sparsity(model)
    
    print("-" * 50)
    print(f"** [FINAL RESULT] Total Sparsity: {final_sparsity:.4f}, Final Accuracy: {final_acc:.2f}% **")
    # 최종적으로 모델을 저장합니다.
    torch.save(model.state_dict(), 'MSPS_Pruned.pth')

if __name__ == "__main__":
    set_seed(2024)

    model = MultiscaleMixer()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load('./checkpoints/MSPS_Mixer_128_22_20251029_173655.pth', map_location=device))
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    criterion = nn.CrossEntropyLoss()

    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((224, 224)),
        torchvision.transforms.ToTensor(),
    ])

    train_data = torchvision.datasets.ImageFolder(root='./data/IAA/train', transform=transform)
    test_data = torchvision.datasets.ImageFolder(root='./data/IAA/test', transform=transform)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=32, shuffle=False, num_workers=2)

    run_iterative_pruning(model, train_loader, test_loader, optimizer, criterion, device)