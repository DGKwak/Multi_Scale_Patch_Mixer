import random
import os
import datetime
import hydra
from hydra.utils import instantiate
from hydra.core.hydra_config import HydraConfig
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, ExponentialLR

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

from model.CSM_New_residual import MultiscaleMixer
from utils.earlystopping import EarlyStopping
from utils.logger import create_logger

def make_datasets(tr_transform,
                  v_transform,
                  train_dir,
                  val_dir,
                  test_dir,):
    # 데이터셋 로드
    train_transform = instantiate(tr_transform)
    val_transform = instantiate(v_transform)

    train_dataset = datasets.ImageFolder(root=train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(root=val_dir, transform=val_transform)
    test_dataset = datasets.ImageFolder(root=test_dir, transform=val_transform)

    return train_dataset, val_dataset, test_dataset

def make_dataloaders(train_dataset,
                     val_dataset,
                     test_dataset,
                     num_workers,
                     batch_size,
                     random_state):

    generator = torch.Generator()
    generator.manual_seed(random_state)

    # DataLoader
    train_loader = DataLoader(train_dataset,
                              num_workers=num_workers,
                              batch_size=batch_size,
                              shuffle=True,
                              generator=generator)
    
    val_loader = DataLoader(val_dataset,
                            num_workers=num_workers,
                            batch_size=batch_size,
                            shuffle=False)
    
    test_loader = DataLoader(test_dataset,
                             num_workers=num_workers,
                             batch_size=batch_size,
                             shuffle=False)
    
    return train_loader, val_loader, test_loader

def set_seed(seed_value):
    random.seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    np.random.seed(seed_value)
    
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # torch.use_deterministic_algorithms(True)

def train(model, 
          loader, 
          optimizer, 
          cross_entropy,
          lambda_aux,
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
        logit, z = model(x)

        ce_loss = cross_entropy(logit, y)
        aux_loss = 0

        for out in z:
            if out.ndim > 2:
                out = torch.mean(out, dim=2, keepdim=False)
            
            aux_loss += cross_entropy(out, y)

        if len(z) == 1:
            aux_loss = 0

        loss = ce_loss + lambda_aux * aux_loss
        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss.item() * x.size(0)
        correct += (logit.argmax(1) == y).sum().item()
    
    return total_loss / len(loader.dataset), correct / len(loader.dataset)

def evaluate(model, 
             loader, 
             cross_entropy,
             lambda_aux, 
             device):
    model.eval()

    total_loss = 0.0
    correct = 0

    with torch.inference_mode():
        for x, y in loader:
            if y.ndim == 2:
                y = torch.argmax(y, dim=1)
            
            x, y = x.to(device), y.to(device)

            logit, z = model(x)

            correct += (logit.argmax(1) == y).sum().item()

            ce_loss = cross_entropy(logit, y)
            aux_loss = 0

            for out in z:
                if out.ndim > 2:
                    out = torch.mean(out, dim=2, keepdim=False)
            
                aux_loss += cross_entropy(out, y)

            if len(z) == 1:
                aux_loss = 0

            loss = ce_loss + lambda_aux * aux_loss

            total_loss += loss.item() * x.size(0)

    return total_loss / len(loader.dataset), correct / len(loader.dataset)

def test(model,
         loader,
         device):
    
    model.eval()

    model.eval()
    
    all_predictions = []
    all_targets = []
    correct = 0
    total = 0
    
    with torch.inference_mode():
        for x, y in tqdm(loader, desc="Testing"):
            if y.ndim == 2:
                y = torch.argmax(y, dim=1)
            
            x, y = x.to(device), y.to(device)
            
            logits, z = model(x)
            predictions = logits.argmax(1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(y.cpu().numpy())
            
            correct += (predictions == y).sum().item()
            total += y.size(0)
    
    test_accuracy = correct / total
    return test_accuracy, all_predictions, all_targets

def plot_confusion_matrix(y_true, y_pred, class_names, experiment_name, save_path):
    """Confusion Matrix 플롯 생성 및 저장"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix - {experiment_name}')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    
    # 저장 경로 생성
    os.makedirs(save_path, exist_ok=True)
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    cm_path = f"{save_path}/{experiment_name}_confusion_matrix_{timestamp}.png"
    
    plt.tight_layout()
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    
    return cm_path

@hydra.main(config_path='./config', config_name='config', version_base=None)
def main(cfg):
    metadata = {
        'Experiment Name': cfg.experiment_name,
        'Dataset': cfg.data.dataset_name,
        'Input_size': cfg.data.input_size,
        'Random_state': cfg.random_state,
        'Epochs': cfg.epochs,
        'Batch_size': cfg.batch_size,
        'Learning_rate': cfg.learning_rate,
        'Patch Size': cfg.model.patches,
        'Patch Dim': cfg.model.patch_dim,
        'Dropout': cfg.model.dropout,
        'Layers': cfg.model.num_layers,
        'Activation': cfg.model.activation,
    }

    log_out = HydraConfig.get().runtime.output_dir
    logger = create_logger(log_out, dist_rank=0, name=cfg.experiment_name)

    logger.info(metadata)
    
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    logger.info(f"Using device: {device}")

    train_dataset, val_dataset, test_dataset = make_datasets(cfg.data.train,
                                                             cfg.data.val,
                                                             cfg.data.train_dir,
                                                             cfg.data.val_dir,
                                                             cfg.data.test_dir,)
    
    # 데이터셋 정보 출력
    logger.info(f"총 데이터셋 크기: {len(train_dataset)}")
    logger.info(f"검증 데이터셋 크기: {len(val_dataset)}")
    logger.info(f"클래스 수: {len(train_dataset.classes)}")
    logger.info(f"클래스 목록: {train_dataset.classes}")
    
    train_loader, val_loader, test_loader = make_dataloaders(train_dataset,
                                                            val_dataset,
                                                            test_dataset,
                                                            cfg.num_workers,
                                                            cfg.batch_size,
                                                            cfg.data.random_state)
    
    set_seed(cfg.random_state)
    
    # Model
    model = MultiscaleMixer(
        in_channels=cfg.model.in_channels,
        patch_dim=cfg.model.patch_dim,
        num_layers=cfg.model.num_layers,
        dropout=cfg.model.dropout,
        patches=cfg.model.patches,
        act=cfg.model.activation,
    ).to(device)

    # loss Function
    cross_entropy = nn.CrossEntropyLoss()

    # Optimizer & Scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.epochs, eta_min=1e-6)
    # scheduler = ExponentialLR(optimizer, gamma=0.9)
    
    early_stopping = EarlyStopping(patience=30, mode='min', verbose=True)

    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

    # Track best validation loss overall
    best_val_loss = float('inf')
    best_model_path = f'{cfg.best_model_path}/{cfg.experiment_name}_{timestamp}.pth'

    train_accuracies, val_accuracies = [], []
    train_losses, val_losses = [], []

    results = []

    for epoch in tqdm(range(cfg.epochs)):
        train_loss, train_acc = train(model,
                                      train_loader,
                                      optimizer,
                                      cross_entropy,
                                      cfg.lambda_aux,
                                      device,
                                      scheduler)
        
        val_loss, val_acc = evaluate(model,
                                     val_loader,
                                     cross_entropy,
                                     cfg.lambda_aux,
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
            logger.info(f"Epoch {epoch + 1}: 검증 손실이 감소했습니다. 최적의 모델을 {best_model_path}에 저장했습니다.")

        logger.info(f"lr : {scheduler.get_last_lr()[0]} \n Epoch {epoch + 1}/{cfg.epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        early_stopping(val_loss)
        if early_stopping.early_stop:
            logger.info("Early stopping triggered. Training halted.")
            break

    # 결과를 DataFrame으로 변환하여 CSV로 저장
    results_df = pd.DataFrame(results)
    os.makedirs(cfg.csv_path, exist_ok=True)

    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    results_csv_path = f"{cfg.csv_path}/{cfg.experiment_name}_results_{timestamp}.csv"

    with open(results_csv_path, 'w') as f:
        for key, value in metadata.items():
            f.write(f"# {key}: {value}\n")
        f.write("\n")
        results_df.to_csv(f, index=False)
    
    logger.info(f"Results saved to {results_csv_path}")

    # 요약 통계 출력
    logger.info(f"\n=== 학습 완료 요약 ===")
    logger.info(f"최고 훈련 정확도: {max(train_accuracies):.4f}")
    logger.info(f"최고 검증 정확도: {max(val_accuracies):.4f}")
    logger.info(f"최종 훈련 손실: {train_losses[-1]:.4f}")
    logger.info(f"최종 훈련 정확도: {train_accuracies[-1]:.4f}")
    logger.info(f"최종 검증 정확도: {val_accuracies[-1]:.4f}")

    # 최적 모델 로드
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        logger.info(f"최적 모델 로드 완료: {best_model_path}")
        
        # 테스트 수행
        test_accuracy, predictions, targets = test(model, test_loader, device)
        
        logger.info(f"테스트 정확도: {test_accuracy:.4f}")
        
        # Classification Report 출력
        logger.info(f"\n=== Classification Report ===")
        logger.info(classification_report(targets, predictions))

        # Confusion Matrix 생성 및 저장
        plot_save_path = cfg.confusion_path
        cm_path = plot_confusion_matrix(targets, predictions, train_dataset.classes, 
                                        cfg.experiment_name, plot_save_path)
        
        logger.info(f"Confusion matrix saved to {cm_path}")
        
        # 테스트 결과를 메타데이터에 추가하여 저장
        test_metadata = metadata.copy()
        test_metadata.update({
            'Test Accuracy': test_accuracy,
            'Best Validation Loss': best_val_loss,
            'Best Model Path': best_model_path,
            'Confusion Matrix Path': cm_path
        })
        
        # 테스트 결과 CSV 저장
        test_results_path = f"{cfg.csv_path}/{cfg.experiment_name}_test_results_{timestamp}.csv"
        with open(test_results_path, 'w') as f:
            for key, value in test_metadata.items():
                f.write(f"# {key}: {value}\n")
            f.write("\n")
            # 클래스별 정확도도 저장
            f.write("# Classification Report\n")
            f.write(classification_report(targets, predictions, target_names=train_dataset.classes))

        logger.info(f"Test results saved to {test_results_path}")

    else:
        logger.info(f"최적 모델 파일을 찾을 수 없습니다: {best_model_path}")

if __name__ == '__main__':
    main()
