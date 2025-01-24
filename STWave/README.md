<div align="center">
  <img src="assets/Basic-TS-logo-for-white.png#gh-light-mode-only" height=200>
  <img src="assets/Basic-TS-logo-for-black.png#gh-dark-mode-only" height=200>
  <h3><b> A Fair and Scalable Time Series Forecasting Benchmark and Toolkit. </b></h3>
</div>

<div align="center">

[**English**](./README.md) **|** 
[**简体中文**](./README_CN.md)

</div>

---

<div align="center">

[![EasyTorch](https://img.shields.io/badge/Developing%20with-EasyTorch-2077ff.svg)](https://github.com/cnstark/easytorch)
[![LICENSE](https://img.shields.io/github/license/zezhishao/BasicTS.svg)](https://github.com/zezhishao/BasicTS/blob/master/LICENSE)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.10.0-orange)](https://pytorch.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.3.1-orange)](https://pytorch.org/)
[![python lint](https://github.com/zezhishao/BasicTS/actions/workflows/pylint.yml/badge.svg)](https://github.com/zezhishao/BasicTS/blob/master/.github/workflows/pylint.yml)

</div>

<div align="center">

🎉 [**Getting Started**](./tutorial/getting_started.md) **|** 
💡 [**Overall Design**](./tutorial/overall_design.md)

📦 [**Dataset**](./tutorial/dataset_design.md) **|** 
🛠️ [**Scaler**](./tutorial/scaler_design.md) **|** 
🧠 [**Model**](./tutorial/model_design.md) **|** 
📉 [**Metrics**](./tutorial/metrics_design.md) **|** 
🏃‍♂️ [**Runner**](./tutorial/runner_design.md) **|** 
📜 [**Config**](./tutorial/config_design.md.md) **|** 
📜 [**Baselines**](./baselines/)

</div>


This is the official repository of BasicTS, which includes the STWave model.  

We have embedded STSGL into STWave. Please follow the guidelines provided in BasicTS to set up the environment and run the code.


## 🚀 Installation and Quick Start

For detailed instructions, please refer to the [Getting Started](./tutorial/getting_started.md) tutorial.

## How to Run STWave
Run STWave using the `STWave/experiments/train.py` file.  
The parameters for each dataset can be modified in the series of files located under `baselines/STWave/PEMS0X.py`.



