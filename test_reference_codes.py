import os
import datetime
import hydra
from hydra.utils import instantiate
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from torch.optim.lr_scheduler import CosineAnnealingLR, ExponentialLR
from sklearn.model_selection import train_test_split

def train(model, 
          loader, 
          optimizer, 
          cross_entropy, 
          device, 
          scheduler):
    model.train()

    total_loss = 0.0
    correct = 0

    for x, y in loader:
        if y.ndim == 2:
            y = torch.argmax(y, dim=1)
        
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        logit = model(x)

        ce_loss = cross_entropy(logit, y)
        loss = ce_loss
        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss.item() * x.size(0)
        correct += (logit.argmax(1) == y).sum().item()
    
    return total_loss / len(loader.dataset), correct / len(loader.dataset)

def evaluate(model, 
             loader, 
             cross_entropy, 
             device):
    model.eval()

    total_loss = 0.0
    correct = 0

    with torch.inference_mode():
        for x, y in loader:
            if y.ndim == 2:
                y = torch.argmax(y, dim=1)
            
            x, y = x.to(device), y.to(device)

            logit = model(x)

            correct += (logit.argmax(1) == y).sum().item()

            ce_loss = cross_entropy(logit, y)
            loss = ce_loss

            total_loss += loss.item() * x.size(0)

    return total_loss / len(loader.dataset), correct / len(loader.dataset)

@hydra.main(config_path='./config', config_name='reference_test')
def main(cfg):
    metadata = {
        'Model': cfg.model_name,
        'Experiment Name': cfg.experiment_name,
        'Dataset': cfg.data.dataset_name,
        'input_size': cfg.data.input_size,
        'random_state': cfg.data.random_state,
        'val_split': cfg.data.val_split,
        'epochs': cfg.epochs,
        'batch_size': cfg.batch_size,
        'learning_rate': cfg.learning_rate,
    }
    
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")

    train_transform = instantiate(cfg.data.train)
    val_transform = instantiate(cfg.data.val)

    train_dataset = datasets.ImageFolder(root=cfg.data.data_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(root=cfg.data.data_dir, transform=val_transform)

    # 데이터셋 정보 출력
    print(f"총 데이터셋 크기: {len(train_dataset)}")
    print(f"검증 데이터셋 크기: {len(val_dataset)*cfg.data.val_split:.0f}")
    print(f"클래스 수: {len(train_dataset.classes)}")
    print(f"클래스 목록: {train_dataset.classes}")
    
    indices = list(range(len(train_dataset)))
    label_list = train_dataset.targets

    train_indices, val_indices, _, _ = train_test_split(
        indices, label_list, test_size=cfg.data.val_split, random_state=42, stratify=label_list
    )

    train_subset = Subset(train_dataset, train_indices)
    val_subset = Subset(val_dataset, val_indices)

    generator = torch.Generator()
    generator.manual_seed(cfg.data.random_state)

    # DataLoader
    train_loader = DataLoader(train_subset,
                              num_workers=cfg.num_workers,
                              batch_size=cfg.batch_size,
                              shuffle=True,)
    
    val_loader = DataLoader(val_subset,
                            num_workers=cfg.num_workers,
                            batch_size=cfg.batch_size,
                            shuffle=False)
    
    # Model
    if cfg.model_name == "HireMLP":
        import Reference_code.Hire_MLP as hire
        
        model = hire.hire_mlp_tiny().to(device)
    elif cfg.model_name == "WaveMLP":
        import Reference_code.wave_mlp as wave
        
        model = wave.WaveMLP_T().to(device)        

    # loss Function
    cross_entropy = nn.CrossEntropyLoss()

    # Optimizer & Scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.epochs, eta_min=1e-6)
    # scheduler = ExponentialLR(optimizer, gamma=0.9)

    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

    # Track best validation loss overall
    best_val_loss = float('inf')
    best_model_path = f'{cfg.best_model_path}/{cfg.experiment_name}_{cfg.model_name}_{timestamp}.pth'

    train_accuracies, val_accuracies = [], []
    train_losses, val_losses = [], []

    results = []

    for epoch in tqdm(range(cfg.epochs)):
        train_loss, train_acc = train(model,
                                      train_loader,
                                      optimizer,
                                      cross_entropy,
                                      device,
                                      scheduler)
        
        val_loss, val_acc = evaluate(model,
                                     val_loader,
                                     cross_entropy,
                                     device)
        
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # 각 epoch 결과를 딕셔너리로 저장
        epoch_result = {
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_accuracy': train_acc,
            'val_loss': val_loss,
            'val_accuracy': val_acc,
        }
        results.append(epoch_result)

        # 최적의 모델 저장 로직
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"Epoch {epoch + 1}: 검증 손실이 감소했습니다. 최적의 모델을 {best_model_path}에 저장했습니다.")

        print(f"lr : {scheduler.get_last_lr()[0]} \n Epoch {epoch + 1}/{cfg.epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

    # 결과를 DataFrame으로 변환하여 CSV로 저장
    results_df = pd.DataFrame(results)
    os.makedirs(cfg.csv_path, exist_ok=True)

    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    results_csv_path = f"{cfg.csv_path}/{cfg.experiment_name}_{cfg.model_name}_results_{timestamp}.csv"

    with open(results_csv_path, 'w') as f:
        for key, value in metadata.items():
            f.write(f"# {key}: {value}\n")
        f.write("\n")
        results_df.to_csv(f, index=False)
    
    print(f"Results saved to {results_csv_path}")

    # 요약 통계 출력
    print(f"\n=== 학습 완료 요약 ===")
    print(f"최고 훈련 정확도: {max(train_accuracies):.4f}")
    print(f"최고 검증 정확도: {max(val_accuracies):.4f}")
    print(f"최종 훈련 손실: {train_losses[-1]:.4f}")
    print(f"최종 훈련 정확도: {train_accuracies[-1]:.4f}")
    print(f"최종 검증 정확도: {val_accuracies[-1]:.4f}")

if __name__ == '__main__':
    main()
