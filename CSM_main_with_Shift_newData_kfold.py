import os
import datetime
import hydra
from hydra.utils import instantiate
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torch.optim.lr_scheduler import CosineAnnealingLR, ExponentialLR

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import StratifiedKFold

# from model.Multi_Scale_Patch_Mixer_ori import MultiscaleMixer
from utils.dataset import Sobel_dataset
# from model.Multi_Scale_Patch_Mixer_with_Shift import MultiscaleMixer
import model.Multi_Scale_Patch_Mixer_with_Shift as ShiftMixer
from utils.earlystopping import EarlyStopping

def make_datasets(train_dir,
                  test_dir,):
    # 데이터셋 로드
    train_dataset = Sobel_dataset(train_dir, is_train=True)
    test_dataset = Sobel_dataset(test_dir, is_train=False)

    # 데이터셋 정보 출력
    print(f"총 데이터셋 크기: {len(train_dataset)}")
    print(f'훈련 데이터셋 크기: {len(train_dataset)}')
    print(f"클래스 수: {len(train_dataset.get_class())}")
    print(f"클래스 목록: {train_dataset.get_class()}")

    return train_dataset, test_dataset

def make_dataloaders(dataset,
                     indices,
                     num_workers,
                     batch_size,
                     shuffle,
                     random_state):
    subset = Subset(dataset, indices)

    generator = None
    if shuffle and random_state:
        generator = torch.Generator()
        generator.manual_seed(random_state)
    
    loader = DataLoader(subset,
                        batch_size=batch_size,
                        shuffle=shuffle,
                        num_workers=num_workers,
                        generator=generator)
    
    return loader

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
    plt.show()
    
    print(f"Confusion matrix saved to {cm_path}")
    return cm_path

@hydra.main(config_path='./config', config_name='CSM_config')
def main(cfg):
    metadata = {
        'Experiment Name': cfg.experiment_name,
        'Dataset': cfg.data.dataset_name,
        'input_size': cfg.data.input_size,
        'random_state': cfg.data.random_state,
        'epochs': cfg.epochs,
        'batch_size': cfg.batch_size,
        'learning_rate': cfg.learning_rate,
        'layers': cfg.model.num_layers,
        'Activation': cfg.model.activation,
        'Patch Size': cfg.model.patches,
        'Dropout': cfg.model.dropout,
    }

    print(metadata)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    train_dataset, test_dataset = make_datasets(cfg.data.train_dir,
                                                 cfg.data.test_dir)
    
    kfold_size = cfg.data.k_fold
    kfold = StratifiedKFold(n_splits=kfold_size, shuffle=True, random_state=cfg.data.random_state)

    kfold_results, fold_summaries = [], []

    for fold, (train_ids, val_ids) in enumerate(kfold.split(np.zeros(len(train_dataset)), train_dataset.get_labels())):
        current_fold = fold+1
        print(f"\n=== K-Fold 교차 검증: Fold {current_fold}/{kfold_size} ===")

        train_loader = make_dataloaders(train_dataset, 
                                        train_ids, 
                                        num_workers=cfg.num_workers, 
                                        batch_size=cfg.batch_size, 
                                        shuffle=True, 
                                        random_state=cfg.data.random_state)
        
        val_loader = make_dataloaders(train_dataset, 
                                      val_ids, 
                                      num_workers=cfg.num_workers, 
                                      batch_size=cfg.batch_size, 
                                      shuffle=False, 
                                      random_state=None)
        
        if cfg.test_model_name == 'test_model_768_8_2_01':
            model = ShiftMixer.test_model_768_8_2_01().to(device)
        elif cfg.test_model_name == 'test_model_768_8_2_02':
            model = ShiftMixer.test_model_768_8_2_02().to(device)
        elif cfg.test_model_name == 'test_model_768_4_2_01':
            model = ShiftMixer.test_model_768_4_2_01().to(device)
        elif cfg.test_model_name == 'test_model_768_4_2_02':
            model = ShiftMixer.test_model_768_4_2_02().to(device)

        elif cfg.test_model_name == 'test_model_128_8_2_01':
            model = ShiftMixer.test_model_128_8_2_01().to(device)
        elif cfg.test_model_name == 'test_model_128_8_2_02':
            model = ShiftMixer.test_model_128_8_2_02().to(device)
        elif cfg.test_model_name == 'test_model_128_4_2_01':
            model = ShiftMixer.test_model_128_4_2_01().to(device)
        elif cfg.test_model_name == 'test_model_128_4_2_02':
            model = ShiftMixer.test_model_128_4_2_02().to(device)
        
        elif cfg.test_model_name == 'horizontal_sliding':
            model = ShiftMixer.horizontal_sliding().to(device)
        
        # loss Function
        cross_entropy = nn.CrossEntropyLoss()

        # Optimizer & Scheduler
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
        scheduler = CosineAnnealingLR(optimizer, T_max=cfg.epochs, eta_min=1e-6)
        # scheduler = ExponentialLR(optimizer, gamma=0.9)
        
        early_stopping = EarlyStopping(patience=60, mode='min', verbose=True)

        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        fold_model_path = f'{cfg.best_model_path}/{cfg.experiment_name}_fold{current_fold}_{timestamp}.pth'

        best_val_loss = float('inf')
        best_val_acc = 0.0

        current_fold_results = []

        for epoch in tqdm(range(cfg.epochs), desc=f"Fold {current_fold} Training"):
            train_loss, train_acc = train(model, train_loader, optimizer, cross_entropy, cfg.loss.lambda_aux, device, scheduler)
            val_loss, val_acc = evaluate(model, val_loader, cross_entropy, cfg.loss.lambda_aux, device)

            epoch_result = {
                'fold': current_fold,
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'train_accuracy': train_acc,
                'val_loss': val_loss,
                'val_accuracy': val_acc,
            }
            current_fold_results.append(epoch_result)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_acc = val_acc
                torch.save(model.state_dict(), fold_model_path)
                print(f"Epoch {epoch + 1}: 검증 손실이 감소했습니다. 최적의 모델을 {fold_model_path}에 저장했습니다.")
            
            print(f"lr : {scheduler.get_last_lr()[0]} \n Epoch {epoch + 1}/{cfg.epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        # K-Fold 결과 저장
        kfold_results.extend(current_fold_results)

        # 각 Fold 요약 통계 저장
        fold_summaries.append({
            'fold': current_fold,
            'best_val_loss': best_val_loss,
            'best_val_accuracy': best_val_acc,
            'fold_results': current_fold_results,
            'best_model_path': fold_model_path
        })

        print(f"Fold {current_fold} 완료: Best Val Loss: {best_val_loss:.4f}, Best Val Acc: {best_val_acc:.4f}")

    # K-Fold 결과 DataFrame 변환 및 CSV 저장
    kfold_results_df = pd.DataFrame(kfold_results)
    os.makedirs(cfg.csv_path, exist_ok=True)

    final_timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    kfold_results_csv_path = f"{cfg.csv_path}/{cfg.experiment_name}_kfold_results_{final_timestamp}.csv"

    with open(kfold_results_csv_path, 'w') as f:
        for key, value in metadata.items():
            f.write(f"# {key}: {value}\n")
        f.write("\n")
        pd.DataFrame(fold_summaries).to_csv(f, index=False)
        f.write("\n# Epoch-wise Results\n")
        kfold_results_df.to_csv(f, index=False)
    
    print(f"K-Fold results saved to {kfold_results_csv_path}")

    # K-Fold 전체 요약
    avg_best_val_acc = pd.DataFrame(fold_summaries)['best_val_accuracy'].mean()
    print(f"\n=== K-Fold 학습 완료 요약 ===")
    print(f"평균 최고 검증 정확도 ({kfold_size} Folds): {avg_best_val_acc:.4f}")

    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    results_csv_path = f"{cfg.csv_path}/{cfg.experiment_name}_results_{timestamp}.csv"

    best_fold_summary = pd.DataFrame(fold_summaries).loc[pd.DataFrame(fold_summaries)['best_val_loss'].idxmin()]
    best_model_to_test_path = best_fold_summary['best_model_path']
    best_fold_loss_row = best_fold_summary['fold']
    best_val_loss_overall = best_fold_summary['best_val_loss']

    # test_loader 생성
    test_loader = make_dataloaders(test_dataset, range(len(test_dataset)), cfg.num_workers, cfg.batch_size, False)

    # 최적 모델 로드
    if os.path.exists(best_model_to_test_path):
        model.load_state_dict(torch.load(best_model_to_test_path, map_location=device))
        print(f"최적 모델 로드 완료: {best_model_to_test_path}")

        # 테스트 수행
        test_accuracy, predictions, targets = test(model, test_loader, device)
        
        print(f"테스트 정확도: {test_accuracy:.4f}")
        
        # Classification Report 출력
        print(f"\n=== 최종 Classification Report ===")
        class_names = test_dataset.get_class()
        report = classification_report(targets, predictions, target_names=class_names, output_dict=True)
        print(classification_report(targets, predictions, target_names=class_names))
        
        # Confusion Matrix 생성 및 저장
        plot_save_path = cfg.confusion_path
        cm_path = plot_confusion_matrix(targets, predictions, train_dataset.get_class(), 
                                        cfg.experiment_name, plot_save_path)
        
        # 테스트 결과를 메타데이터에 추가하여 저장
        test_metadata = metadata.copy()
        test_metadata.update({
            'K-Fold Avg Best Val Acc': avg_best_val_acc,
            'Best Val Loss Overall': best_val_loss_overall,
            'Best Fold for Test': best_fold_loss_row,
            'Final Test Accuracy': test_accuracy,
            'Best Model Path': best_model_to_test_path,
            'Confusion Matrix Path': cm_path
        })
        
        # 최종 테스트 결과 CSV 저장
        test_results_path = f"{cfg.csv_path}/{cfg.experiment_name}_final_test_results_{final_timestamp}.csv"
        with open(test_results_path, 'w') as f:
            for key, value in test_metadata.items():
                f.write(f"# {key}: {value}\n")
            f.write("\n")
            # Classification Report도 저장
            f.write("# Classification Report\n")
            pd.DataFrame(report).transpose().to_csv(f, index=True)
            
        print(f"Final Test results saved to {test_results_path}")
        
    else:
        print(f"최적 모델 파일을 찾을 수 없습니다: {best_model_to_test_path}")

if __name__ == '__main__':
    main()
