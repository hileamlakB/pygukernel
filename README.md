# pygukernel

# pygukernel

## 📖 Project Overview
This repository contains a Python-based project focused on benchmarking and optimizing kernel operations using PyTorch and Triton. The project is designed to evaluate the performance of various kernel operations, such as matrix multiplication and ReLU activation, by comparing their execution times and memory usage. The target users include machine learning engineers and researchers who are interested in optimizing deep learning models for performance.

## 🏗️ Technical Architecture
The project is structured around two main components: a benchmarking module and a kernel optimization module. The benchmarking module (`benchmarking_clone.py`) captures and logs performance metrics, while the kernel optimization module (`dynamo_inductor.py`) leverages PyTorch's Dynamo and Inductor to compile and optimize kernel operations. The interaction between these components is facilitated through the use of Triton kernels for GPU acceleration.

## 📁 File Documentation

### 📄 benchmarking_clone.py
This file contains the `Benchmark` class, which is responsible for capturing performance metrics, logging results, and generating CSV files for analysis. It handles the execution of Triton kernels and records the compute time, memory reads, and writes.

### 📄 dynamo_inductor.py
This file focuses on optimizing kernel operations using PyTorch's Dynamo and Inductor. It compiles models, runs benchmarks, and logs results into a structured directory. The file also includes test models for benchmarking purposes.

### 📄 results/index.txt
This file keeps track of the number of benchmark runs by storing an index that increments with each execution. It ensures that each benchmark run is stored in a unique directory.

### 📄 results/result_*/benchmark_*.csv
These CSV files store the detailed results of each benchmark run, including node types, inputs, outputs, and performance metrics such as compute time, memory reads, and writes.

### 📄 results/result_*/original_code_*.txt
These files contain the original code of the models being benchmarked. They provide a reference for the unoptimized code that is used in the benchmarking process.

### 📄 results/result_*/triton_code_*.py
These files contain the Triton-optimized kernel code generated during the benchmarking process. They are used to execute the optimized operations on the GPU.

## 🔧 Installation
To install the required dependencies, run the following commands:

```bash
pip install torch
pip install triton
```

## 🚀 Usage
To run a benchmark, execute the following Python code:

```python
from dynamo_inductor import test_model

# Define a simple model
model = torch.nn.Sequential(
    torch.nn.Linear(3, 10),
    torch.nn.ReLU()
).cuda()

# Run the benchmark
test_model(model, inputs=[torch.randn(100, 3).cuda()])
```

## 📋 Requirements
- Python 3.8 or higher
- PyTorch 2.0 or higher
- Triton 2.0 or higher
- CUDA-enabled GPU

To install the specific versions of the dependencies, use the following commands:

```bash
pip install torch==2.0.0
pip install triton==2.0.0
```

## 📝 License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.