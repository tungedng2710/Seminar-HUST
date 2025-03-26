# Optimizers
Final Test Accuracy Comparison:
SGD: 92.82%
Adam: 98.88%
RMSprop: 97.67%
LBFGS: 98.04%

# Knowledge distillation
Final Test Results:
Teacher Model (ResNet-18)   - Loss: 0.2528, Accuracy: 93.07%
Distilled Student (LeNet-5) - Loss: 1.1914, Accuracy: 64.01%
Scratch Student (LeNet-5)   - Loss: 1.3609, Accuracy: 62.64%

# Quantization
Using device: cuda
[FP32 ] time: 0.158584 s
[FP16 ] time: 0.263971 s | mean abs diff vs FP32: 0.009713763371109962
[BF16 ] time: 0.175937 s | mean abs diff vs FP32: 0.07238982617855072
[INT8 ] time: 0.000523 s | mean abs diff vs FP32: 17.985334396362305