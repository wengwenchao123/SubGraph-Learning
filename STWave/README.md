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

$\text{BasicTS}^{+}$ (**Basic** **T**ime **S**eries) is a benchmark library and toolkit designed for time series forecasting. It now supports a wide range of tasks and datasets, including spatial-temporal forecasting and long-term time series forecasting. It covers various types of algorithms such as statistical models, machine learning models, and deep learning models, making it an ideal tool for developing and evaluating time series forecasting models.

If you find this project helpful, please don't forget to give it a ⭐ Star to show your support. Thank you!

On one hand, BasicTS provides a **unified and standardized pipeline**, offering a **fair and comprehensive** platform for reproducing and comparing popular models.

On the other hand, BasicTS offers a **user-friendly and easily extensible** interface, enabling quick design and evaluation of new models. Users can simply define their model structure and easily perform basic operations.

You can find detailed tutorials in [Getting Started](./tutorial/getting_started.md). Additionally, we are collecting **ToDo** and **HowTo** items. If you need more features (e.g., additional datasets or benchmark models) or tutorials, feel free to open an issue or leave a comment [here](https://github.com/zezhishao/BasicTS/issues/95).


> [!IMPORTANT]  
> If you find this repository helpful for your work, please consider citing the following benchmarking paper:
> ```LaTeX
> @article{shao2023exploring,
>    title={Exploring Progress in Multivariate Time Series Forecasting: Comprehensive Benchmarking and Heterogeneity Analysis},
>    author={Shao, Zezhi and Wang, Fei and Xu, Yongjun and Wei, Wei and Yu, Chengqing and Zhang, Zhao and Yao, Di and Jin, Guangyin and Cao, Xin and Cong, Gao and others},
>    journal={arXiv preprint arXiv:2310.06119},
>    year={2023}
>  }
>  ```
> 🔥🔥🔥 ***The paper has been accepted by IEEE TKDE! You can check it out [here](https://arxiv.org/abs/2310.06119).***  🔥🔥🔥


## ✨ Highlighted Features

### Fair Performance Review

Users can compare the performance of different models on arbitrary datasets fairly and exhaustively based on a unified and comprehensive pipeline.

### Developing with BasicTS

<details>
  <summary><b>Minimum Code</b></summary>
Users only need to implement key codes such as model architecture and data pre/post-processing to build their own deep learning projects.
</details>

<details>
  <summary><b>Everything Based on Config</b></summary>
Users can control all the details of the pipeline through a config file, such as the hyperparameter of dataloaders, optimization, and other tricks (*e.g.*, curriculum learning). 
</details>

<details>
  <summary><b>Support All Devices</b></summary>
BasicTS supports CPU, GPU and GPU distributed training (both single node multiple GPUs and multiple nodes) thanks to using EasyTorch as the backend. Users can use it by setting parameters without modifying any code.
</details>

<details>
  <summary><b>Save Training Log</b></summary>
Support `logging` log system and `Tensorboard`, and encapsulate it as a unified interface, users can save customized training logs by calling simple interfaces.
</details>

## 🚀 Installation and Quick Start

For detailed instructions, please refer to the [Getting Started](./tutorial/getting_started.md) tutorial.

