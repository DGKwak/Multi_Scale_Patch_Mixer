import os
import random
from typing import Dict, Any, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms

import onnx
import onnxruntime as ort

import time
import copy
import numpy as np

from model.MSPS_Mixer_for_quant import MultiscaleMixer, MlpBlock

def set_random_seeds(random_seed=0):

    torch.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)

def prepare_dataloader(num_workers=8, train_batch_size=32, eval_batch_size=32):

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])

    train_set = torchvision.datasets.ImageFolder('./data/IAA/train', transform=train_transform)
    # We will use test set for validation and test in this project.
    # Do not use test set for validation in practice!
    test_set = torchvision.datasets.ImageFolder('./data/IAA/val', transform=test_transform)

    train_sampler = torch.utils.data.RandomSampler(train_set)
    test_sampler = torch.utils.data.SequentialSampler(test_set)

    train_loader = torch.utils.data.DataLoader(
        dataset=train_set, batch_size=train_batch_size,
        sampler=train_sampler, num_workers=num_workers)

    test_loader = torch.utils.data.DataLoader(
        dataset=test_set, batch_size=eval_batch_size,
        sampler=test_sampler, num_workers=num_workers)

    return train_loader, test_loader

def evaluate_model(model, test_loader, device, criterion=None):

    model.eval()
    model.to(device)

    running_loss = 0
    running_corrects = 0

    for inputs, labels in test_loader:

        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)

        if criterion is not None:
            loss = criterion(outputs, labels).item()
        else:
            loss = 0

        # statistics
        running_loss += loss * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    eval_loss = running_loss / len(test_loader.dataset)
    eval_accuracy = running_corrects / len(test_loader.dataset)

    return eval_loss, eval_accuracy

def train_model(model, train_loader, test_loader, device, learning_rate=1e-1, num_epochs=200):

    # The training configurations were not carefully selected.

    criterion = nn.CrossEntropyLoss()

    model.to(device)

    # It seems that SGD optimizer is better than Adam optimizer for ResNet18 training on CIFAR10.
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-4)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=500)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150], gamma=0.1, last_epoch=-1)
    # optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)

    # Evaluation
    model.eval()
    eval_loss, eval_accuracy = evaluate_model(model=model, test_loader=test_loader, device=device, criterion=criterion)
    print("Epoch: {:02d} Eval Loss: {:.3f} Eval Acc: {:.3f}".format(-1, eval_loss, eval_accuracy))

    for epoch in range(num_epochs):

        # Training
        model.train()

        running_loss = 0
        running_corrects = 0

        for inputs, labels in train_loader:

            inputs = inputs.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        train_loss = running_loss / len(train_loader.dataset)
        train_accuracy = running_corrects / len(train_loader.dataset)

        # Evaluation
        model.eval()
        eval_loss, eval_accuracy = evaluate_model(model=model, test_loader=test_loader, device=device, criterion=criterion)

        # Set learning rate scheduler
        scheduler.step()

        print("Epoch: {:03d} Train Loss: {:.3f} Train Acc: {:.3f} Eval Loss: {:.3f} Eval Acc: {:.3f}".format(epoch, train_loss, train_accuracy, eval_loss, eval_accuracy))

    return model

def calibrate_model(model, loader, device=torch.device("cpu:0")):

    model.to(device)
    model.eval()

    for inputs, labels in loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        _ = model(inputs)

def measure_inference_latency(model,
                              device,
                              input_size=(1, 3, 32, 32),
                              num_samples=100,
                              num_warmups=10):

    model.to(device)
    model.eval()

    x = torch.rand(size=input_size).to(device)

    with torch.no_grad():
        for _ in range(num_warmups):
            _ = model(x)
    torch.cuda.synchronize()

    with torch.no_grad():
        start_time = time.time()
        for _ in range(num_samples):
            _ = model(x)
            torch.cuda.synchronize()
        end_time = time.time()
    elapsed_time = end_time - start_time
    elapsed_time_ave = elapsed_time / num_samples

    return elapsed_time_ave

def save_model(model, model_dir, model_filename):

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model_filepath = os.path.join(model_dir, model_filename)
    torch.save(model.state_dict(), model_filepath)

def load_model(model, model_filepath, device):

    model.load_state_dict(torch.load(model_filepath, map_location=device), strict=False)

    return model

def save_onnx_model(model, model_dir, model_filename):

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    model_filepath = os.path.join(model_dir, model_filename)
    dummy_input = torch.randn((1, 3, 224, 224), device='cpu')
    torch.onnx.export(model, 
                      dummy_input, 
                      model_filepath, 
                      export_params=True, 
                      opset_version=12,
                      do_constant_folding=True,
                      input_names = ['input'],
                      output_names = ['output'],
                      dynamo=True)
    
    print(f"✅ ONNX Model saved successfully to: {model_filepath}")

def load_onnx_model(model_filepath, device):
    ort_session = ort.InferenceSession(model_filepath, providers=['CPUExecutionProvider'])
    
    return ort_session

def evaluate_onnx_model(model, test_loader):

    running_corrects = 0
    total_samples = 0

    input_name = model.get_inputs()[0].name
    output_name = model.get_outputs()[0].name

    for inputs, labels in test_loader:
        inputs_np = inputs.cpu().numpy()

        ort_inputs = {input_name: inputs_np}
        ort_outputs = model.run([output_name], ort_inputs)

        output_np = ort_outputs[0]
        preds_np = np.argmax(output_np, axis=1)

        corrects = np.sum(preds_np == labels.cpu().numpy())
        running_corrects += corrects
        total_samples += inputs.size(0)
    
    eval_accuracy = running_corrects / total_samples

    return 0.0, eval_accuracy

def measure_onnx_latency(ort_session: ort.InferenceSession, input_size: Tuple[int, ...], num_samples=100, num_warmups=10) -> float:
    """ONNX Runtime 세션의 CPU 추론 지연 시간을 측정합니다."""
    
    # ONNX Runtime은 NumPy 입력을 기대합니다.
    # PyTorch 텐서 대신 NumPy 텐서를 생성합니다.
    inputs_np = np.random.rand(*input_size).astype(np.float32)
    input_name = ort_session.get_inputs()[0].name
    ort_inputs = {input_name: inputs_np}

    # 워밍업
    for _ in range(num_warmups):
        _ = ort_session.run(None, ort_inputs)

    # 실제 측정
    start_time = time.time()
    for _ in range(num_samples):
        _ = ort_session.run(None, ort_inputs)
    end_time = time.time()
    
    elapsed_time_ave = (end_time - start_time) / num_samples

    return elapsed_time_ave

def create_model():
    model = MultiscaleMixer()

    return model

class QuantizedMultiscaleMixer(nn.Module):
    def __init__(self, model_fp32):
        super().__init__()
        # QuantStub converts tensors from floating point to quantized.
        # This will only be used for inputs.
        self.quant = torch.quantization.QuantStub()
        # DeQuantStub converts tensors from quantized to floating point.
        # This will only be used for outputs.
        self.dequant = torch.quantization.DeQuantStub()
        # FP32 model
        self.model_fp32 = model_fp32

    def forward(self, x):
        # manually specify where tensors will be converted from floating
        # point to quantized in the quantized model
        x = self.quant(x)
        x = self.model_fp32(x)
        # manually specify where tensors will be converted from quantized
        # to floating point in the quantized model
        x = self.dequant(x)
        return x

def model_equivalence(model_1, model_2, device, rtol=1e-05, atol=1e-08, num_tests=100, input_size=(1,3,224,224)):

    model_1.to(device)
    model_2.to(device)

    for _ in range(num_tests):
        x = torch.rand(size=input_size).to(device)
        y1 = model_1(x).detach().cpu().numpy()
        y2 = model_2(x).detach().cpu().numpy()
        if np.allclose(a=y1, b=y2, rtol=rtol, atol=atol, equal_nan=False) == False:
            print("Model equivalence test sample failed: ")
            print(y1)
            print(y2)
            return False

    return True

# def fuse_multiscalemixer_modules(model):
#     for name, module in model.named_children():
#         if isinstance(module, MlpBlock):
#             torch.quantization.fuse_modules(module, ['linear', 'act'], inplace=True)
        
#         elif name == 'excitation' and isinstance(module, torch.nn.Sequential):
#             torch.quantization.fuse_modules(module, ['0', '1'], inplace=True)
        
#         elif len(list(module.named_children())) > 0:
#             fuse_multiscalemixer_modules(module)

def main():

    random_seed = 2024
    cuda_device = torch.device("cuda:0")
    cpu_device = torch.device("cpu:0")

    model_dir = "./checkpoints"
    model_filename = "MSPS_for_quant04_20251104_131521.pth"
    quantized_model_filename = "MSPS_rev_quantized.onnx"
    model_filepath = os.path.join(model_dir, model_filename)
    quantized_model_filepath = os.path.join(model_dir, quantized_model_filename)

    set_random_seeds(random_seed=random_seed)

    train_loader, test_loader = prepare_dataloader(num_workers=8, train_batch_size=32, eval_batch_size=32)
    
    # Create an untrained model.
    model = create_model()

    # ① floating point 타입으로 모델을 학습하거나 pre-trained 모델을 불러옵니다.
    # Load a pretrained model.
    model = load_model(model=model, model_filepath=model_filepath, device=cuda_device)
    # Move the model to CPU since static quantization does not support CUDA currently.
    
    # ② 모델을 CPU 상태로 두고 학습 모드로 변환합니다. (model.train())
    model.to(cpu_device)
    # Make a copy of the model for layer fusion
    # fused_model = copy.deepcopy(model)

    model.train()
    # The model has to be switched to training mode before any layer fusion.
    # Otherwise the quantization aware training will not work correctly.
    # fused_model.train()
    
    # ③ layer fusion을 적용합니다.
    # Fuse the model in place rather manually.
    # fused_model = torch.quantization.fuse_modules(fused_model, [["conv1", "bn1", "relu"]], inplace=True)
    # for module_name, module in fused_model.named_children():
    #     if "layer" in module_name:
    #         for basic_block_name, basic_block in module.named_children():
    #             torch.quantization.fuse_modules(basic_block, [["conv1", "bn1", "relu1"], ["conv2", "bn2"]], inplace=True)
    #             for sub_block_name, sub_block in basic_block.named_children():
    #                 if sub_block_name == "downsample":
    #                     torch.quantization.fuse_modules(sub_block, [["0", "1"]], inplace=True)

    # fuse_multiscalemixer_modules(fused_model)

    # with open('FP32_model.txt', 'w') as f:
    #     f.write('=== FP32 Model ===\n')
    #     f.write(str(model))

    # with open('Fused_model.txt', 'w') as f:
    #     f.write('=== Fused Model ===\n')
    #     f.write(str(fused_model))

    # Print FP32 model.
    # print(model)
    # Print fused model.
    # print(fused_model)

    # ④ 모델을 평가 모드로 변환 후 (model.eval()) layer fusion이 잘 적용되었는 지 확인합니다. 확인 후에는 다시 학습 모드로 변경해 줍니다.
    # Model and fused model should be equivalent.
    model.eval()
    # fused_model.eval()
    # assert model_equivalence(model_1=model, model_2=fused_model, device=cpu_device, rtol=1e-03, atol=1e-06, num_tests=100, input_size=(1,3,224,224)), "Fused model is not equivalent to the original model!"

    
    # ⑤ input에는 torch.quantization.QuantStub()를 적용시키고 output에는 torch.quantization.DeQuantStub()을 적용시킵니다.
    # Prepare the model for quantization aware training. This inserts observers in
    # the model that will observe activation tensors during calibration.
    quantized_model = QuantizedMultiscaleMixer(model_fp32=model)
    # Using un-fused model will fail.
    # Because there is no quantized layer implementation for a single batch normalization layer.
    # quantized_model = QuantizedResNet18(model_fp32=model)
    
    # ⑥ quantization configuration을 지정합니다. (ex. symmetric quantization, asymmetric quantization)
    # Select quantization schemes from 
    # https://pytorch.org/docs/stable/quantization-support.html
    quantization_config = torch.quantization.get_default_qconfig("fbgemm")
    # Custom quantization configurations
    # quantization_config = torch.quantization.default_qconfig
    # quantization_config = torch.quantization.QConfig(activation=torch.quantization.MinMaxObserver.with_args(dtype=torch.quint8), weight=torch.quantization.MinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_tensor_symmetric))

    quantized_model.qconfig = quantization_config
    
    # Print quantization configurations
    print(quantized_model.qconfig)

    # https://pytorch.org/docs/stable/_modules/torch/quantization/quantize.html#prepare_qat
    torch.quantization.prepare_qat(quantized_model, inplace=True)

    # ⑦ QAT를 하기 위하여 quantization 모델을 준비합니다.
    # # Use training data for calibration.
    print("Training QAT Model...")
    quantized_model.train()
    
    # ⑧ 모델을 다시 CUDA가 상태로 적용하고 CUDA를 이용하여 QAT를 모델 학습을 진행합니다.
    train_model(model=quantized_model, train_loader=train_loader, test_loader=test_loader, device=cuda_device, learning_rate=1e-3, num_epochs=10)
    
    # ⑨ 모델을 다시 CPU 상태로 두고 QAT가 적용된 floating point 모델을 quantized integer model로 변환합니다.    
    quantized_model.to(cpu_device)    
    # Using high-level static quantization wrapper
    # The above steps, including torch.quantization.prepare, calibrate_model, and torch.quantization.convert, are also equivalent to
    # quantized_model = torch.quantization.quantize_qat(model=quantized_model, run_fn=train_model, run_args=[train_loader, test_loader, cuda_device], mapping=None, inplace=False)

    # ⑪ quantized integer model을 저장합니다.
    quantized_model = torch.quantization.convert(quantized_model, inplace=True)

    quantized_model.eval()

    with open('Quantized_model.txt', 'w') as f:
        f.write('=== Quantized Model ===\n')
        f.write(str(quantized_model))
    # Print quantized model.
    # print(quantized_model)

    # Save quantized model.
    save_onnx_model(model=quantized_model, model_dir=model_dir, model_filename=quantized_model_filename)

    # Load quantized model.
    quantized_onnx_model = load_onnx_model(model_filepath=quantized_model_filepath, device=cpu_device)

    _, fp32_eval_accuracy = evaluate_model(model=model, test_loader=test_loader, device=cpu_device, criterion=None)
    _, int8_eval_accuracy = evaluate_onnx_model(model=quantized_onnx_model, test_loader=test_loader)

    # Skip this assertion since the values might deviate a lot.
    # assert model_equivalence(model_1=model, model_2=quantized_onnx_model, device=cpu_device, rtol=1e-01, atol=1e-02, num_tests=100, input_size=(1,3,224,224)), "Quantized model deviates from the original model too much!"

    print("FP32 evaluation accuracy: {:.3f}".format(fp32_eval_accuracy))
    print("INT8 evaluation accuracy: {:.3f}".format(int8_eval_accuracy))

    fp32_cpu_inference_latency = measure_inference_latency(model=model, device=cpu_device, input_size=(1,3,224,224), num_samples=100)
    int8_cpu_inference_latency = measure_inference_latency(model=quantized_model, device=cpu_device, input_size=(1,3,224,224), num_samples=100)
    int8_onnx_cpu_inference_latency = measure_onnx_latency(ort_session=quantized_onnx_model, input_size=(1,3,224,224), num_samples=100)
    fp32_gpu_inference_latency = measure_inference_latency(model=model, device=cuda_device, input_size=(1,3,224,224), num_samples=100)

    print("FP32 CPU Inference Latency: {:.2f} ms / sample".format(fp32_cpu_inference_latency * 1000))
    print("FP32 CUDA Inference Latency: {:.2f} ms / sample".format(fp32_gpu_inference_latency * 1000))
    print("INT8 CPU Inference Latency: {:.2f} ms / sample".format(int8_cpu_inference_latency * 1000))
    print("INT8 ONNX CPU Inference Latency: {:.2f} ms / sample".format(int8_onnx_cpu_inference_latency * 1000))

if __name__ == "__main__":

    main()