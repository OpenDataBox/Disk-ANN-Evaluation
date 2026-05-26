# Disk-ANN-Evaluation

This repository contains the experimental codebase and evaluation scripts for our comprehensive study on disk-resident graph-based approximate nearest neighbor (ANN) search methods.


## Repository Structure

This repository includes implementations and evaluation scripts for the following disk-resident ANN systems:

- **AiSAQ**: A scalable, DRAM-free approximate nearest neighbor search method with product quantization. Features inline PQ vectors, optimal vector rearrangement.
  - Original repository: [https://github.com/antifreeze53/aisaq-diskann](https://github.com/KioxiaAmerica/aisaq-diskann)
  - See [AiSAQ/README.md](AiSAQ/README.md) for detailed documentation

- **FreshDiskANN**: Microsoft's DiskANN implementation with support for dynamic updates and streaming scenarios.
  - Original repository: https://github.com/ShuiXianhua/FreshDISKANN
  - See [FreshDiskANN/README.md](FreshDiskANN/README.md) for detailed documentation

- **Gorgeous**: High-performance disk-based vector search system with optimized data layout for large-scale high-dimensional datasets. This directory also includes implementations of Starling and DiskANN for comparative evaluation.
  - Original repository: https://github.com/yinpeiqi/Gorgeous
  - See [Gorgeous/README.md](Gorgeous/README.md) for detailed documentation

- **PageANN**: Page-level approximate nearest neighbor search system with disk-aligned graph organization to reduce random I/O.
  - Original repository: https://github.com/Dingyi-Kang/PageANN
  - See [PageANN/README.md](PageANN/README.md) for detailed documentation

- **PipeANN**: Low-latency, billion-scale, updatable graph-based vector store with ultra-low latency (<1ms) and efficient update mechanisms.
  - Original repository: https://github.com/thustorage/PipeANN
  - See [PipeANN/README.md](PipeANN/README.md) for detailed documentation
- **DGAI**: Decoupled On-Disk Graph‑Based ANN Index for Efficient Updates and Queries
  - Original repository: https://github.com/iDC-NEU/DGAI
  - See [DGAI/README_run.md](DGAI/README_run.md) for detailed documentation

## Getting Started

### Prerequisites

Each system has its own dependencies. Please refer to the individual README files in each subdirectory for specific requirements. Common dependencies include:

- CMake (v3.15+)
- C++ compiler (g++ or clang)
- Intel MKL
- Boost libraries
- libaio-dev (Linux)
- liburing-dev (for AiSAQ)

### Building

Please refer to the README file in each system's directory for specific build instructions:

- [AiSAQ Build Instructions](AiSAQ/README.md)
- [FreshDiskANN Build Instructions](FreshDiskANN/README.md)
- [Gorgeous Build Instructions](Gorgeous/README.md)
- [PageANN Build Instructions](PageANN/README.md)
- [PipeANN Build Instructions](PipeANN/README.md)
- [DGAI Build Instructions](DGAI/README_run.md)

### Running Experiments

Detailed instructions for reproducing the experiments can be found in each system's documentation. Each system provides scripts and configuration files for running benchmarks and evaluations.

## License

Each system in this repository retains its original license:

- **AiSAQ**: MIT License (Copyright Microsoft Corporation)
- **FreshDiskANN**: MIT License (Copyright Cong Fu, Changxu Wang, Deng Cai)
- **Gorgeous**: MIT License (Copyright Microsoft Corporation)
- **PageANN**: MIT License (Copyright Microsoft Corporation & Dingyi Kang)
- **PipeANN**: MIT License (Copyright Hao Guo)

Please refer to the LICENSE file in each subdirectory for full license text.
