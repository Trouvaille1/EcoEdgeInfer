# EcoEdgeInfer

This repository contains the implementation code for our paper - EcoEdgeInfer ([ACM Digital Library](https://dl.acm.org/doi/10.1109/SEC62691.2024.00023), [PDF](https://pramodh.rachuri.dev/files/pramodh_EcoEdgeInfer_sec24.pdf), [Slides](https://pramodh.rachuri.dev/files/pramodh_EcoEdgeInfer_sec24_slides.pdf)).

## Quick Start

To see the EcoEdgeInfer framework and the **EcoGD optimizer** in action, simply run the `example_EcoGD.py` file:  
```bash
python3 example_EcoGD.py
```

> **Note:** The code is currently being cleaned and better documented for enhanced readability. However, it should function as expected. If you encounter any issues, feel free to raise an issue in this repository.

## Table of Contents
1. [EcoEdgeInfer Framework](#1-ecoedgeinferpy)  
   - [Optimizers](#optimizers)  
   - [Extensibility](#extensibility)  
2. [nvpmplus Library](#2-nvpmpluspy)  
   - [Features](#features)  
   - [Usage](#usage)  
3. [power_profile Library](#3-power_profilepy)  
   - [Features](#features-1)  
   - [Usage](#usage-1)  
4. [Citation](#citation)  
5. [Miscellaneous](#miscellaneous)  
---

### 1. `EcoEdgeInfer.py`
This file contains the implementation of the **EcoEdgeInfer** framework. Import this module to integrate the framework into your own projects. Below is a high-level overview of the framework:

<p align="center">
    <img src="system_design.jpg" alt="EcoEdgeInfer Framework Design" width="70%"/>
</p>

#### Optimizers
The following optimizers are included in the framework:  
- Grid Search  
- Linear Search  
- DVFS  
- Multi-Armed Bandit (Independent Dimensions)  
- **EcoGD**  
- Bayesian Optimization  
- Multi-Armed Bandit (Joint Search Space)  
- Random Choice  
- Fixed Choice  

Details and comparisons for a subset of these optimizers are available in the paper.  

#### Extensibility
To add a custom optimizer:  
1. Extend the `EnergyOptimizer_skeleton` class.  
2. Implement the `run_optimizer` method.  

For reference, you can review the implementation of `EnergyOptimizer_linearsweeps` (a simple optimizer).  
Additionally, method overriding can be used to adjust behavior. For example, the `EnergyOptimizer_MAB_multiDim` class overrides the `update_history` method to modify how history is updated for the MAB optimizer.  

Detailed method and class descriptions are available as **docstrings** within the code.

---

### 2. `nvpmplus.py`
Our **nvpmplus** script offers fine-grained control over power modes on Jetson devices (e.g., Nano, Xavier NX). While `nvpmodel` provides limited preset modes, `nvpmplus` allows setting custom CPU and GPU frequencies as well as changing governors.  

#### Features:
- Adjust CPU and GPU frequencies to any value within the device’s range.  
- Change the governor of the CPU and GPU.  
- Tested on Jetson Nano and Xavier NX.  

#### Usage:
It can used within Python scripts or as a standalone script. For running within a script, import the library and use the `set_state` or `set_gov` method. Example:  `nvpmplus.set_state(nvpmplus.cpu_lim, cpu, gpu)`
For standalone usage, run the script with the desired arguments. More details are available in the help section.

```
> python3 nvpmplus.py --help

running on Jetson Xavier
usage: nvpmplus.py [-h] [--cpus CPUS] [--cpu_max_fq CPU_MAX_FQ] [--gpu_max_fq GPU_MAX_FQ]
                   [--cpu_gov CPU_GOV] [--gpu_gov GPU_GOV] [--ONLY_GOV ONLY_GOV] [--ONLY_FREQ ONLY_FREQ]

nvpmodel plus

options:
  -h, --help            show this help message and exit
  --cpus CPUS           Input can be 1 to 6
  --cpu_max_fq CPU_MAX_FQ
                        Input can be 0 to 24
  --gpu_max_fq GPU_MAX_FQ
                        Input can be 0 to 14
  --cpu_gov CPU_GOV     Input a CPU governor: interactive, conservative, ondemand,
                        userspace, powersave, performance, schedutil
  --gpu_gov GPU_GOV     Input a GPU governor: wmark_simple, nvhost_podgov,
                        userspace, performance, simple_ondemand
  --ONLY_GOV ONLY_GOV   Set only the governor
  --ONLY_FREQ ONLY_FREQ
                        Set only the frequency
```

---

### 3. `power_profile.py`
Our **power_profile** script measures power consumption of specific functions with high precision. Unlike `tegrastats` (which samples every second), `power_profile` offers higher sampling rates, making it easier to measure energy consumption for individual functions.

#### Features:
- Measure energy consumption in Joules.  
- High sampling rate for precise profiling.  
- Tested on Jetson Nano and Xavier NX.

#### Usage:
Use the `energy_calculator` method to measure energy consumption:  
```python
power_profile.energy_calculator(user_function, batch_input)
```
- **`user_function`**: The function to be profiled.  
- **`batch_input`**: The input to the function.  

The method returns the energy consumed in Joules.

---

## Citation

If you use this code or find the **EcoEdgeInfer** framework helpful in your research, please cite our paper:  
```bibtex
@inproceedings{rachuri2024ecoedgeinfer,
author = {Rachuri, Sri Pramodh and Shaik, Nazeer and Choksi, Mehul and Gandhi, Anshul},
title = {EcoEdgeInfer: Dynamically Optimizing Latency and Sustainability for Inference on Edge Devices},
year = {2025},
publisher = {IEEE Press},
url = {https://doi.org/10.1109/SEC62691.2024.00023},
doi = {10.1109/SEC62691.2024.00023},
abstract = {The use of Deep Neural Networks (DNNs) has skyrocketed in recent years. While its applications have brought many benefits and use cases, they also have a significant environmental impact due to the high energy consumption of DNN execution. It has already been acknowledged in the literature that training DNNs is computationally expensive and requires large amounts of energy. However, the energy consumption of DNN inference is still an area that has not received much attention, yet. With the increasing adoption of online tools, the usage of inference has significantly grown and will likely continue to grow. Unlike training, inference is user-facing, requires low latency, and is used more frequently. As such, edge devices are being considered for DNN inference due to their low latency and privacy benefits. In this context, inference on edge is a timely area that requires closer attention to regulate its energy consumption. We present EcoEdgeInfer, a system that balances performance and sustainability for DNN inference on edge devices. Our core component of EcoEdgeInfer is an adaptive optimization algorithm, EcoGD, that strategically and quickly sweeps through the hardware and software configuration space to find the jointly optimal configuration that can minimize energy consumption and latency. EcoGD is agile by design, and adapts the configuration parameters in response to time-varying and unpredictable inference workload. We evaluate EcoEdgeInfer on different DNN models using real-world traces and show that EcoGD consistently outperforms existing baselines, lowering energy consumption by 31% and reducing tail latency by 14%, on average.},
booktitle = {Proceedings of the 2024 IEEE/ACM Symposium on Edge Computing},
pages = {191–205},
numpages = {15},
location = {Rome, Italy},
series = {SEC '24}
}
```

---

## Miscellaneous

Due to size constraints, trace logs and processed data are not included in this repository. If you need access, please contact me using the information provided at [pramodh.rachuri.dev](https://pramodh.rachuri.dev/), and I’ll be happy to provide the data.

---