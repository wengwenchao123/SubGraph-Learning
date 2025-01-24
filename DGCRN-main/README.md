# The LargeST Benchmark Dataset

This is the official repository for the NeurIPS 2023 DB Track paper [LargeST: A Benchmark Dataset for Large-Scale Traffic Forecasting](https://arxiv.org/abs/2306.08259). The DGCRN model is integrated within LargeST, and we have incorporated the SGL module into this framework for experimentation.

## 1. Experiments Running

To run the DGCRN baseline, you may execute the Python file in the terminal:
```
python experiments/DGCRN/main.py --device cuda:2 --dataset PEMSD8 --years 2019 --model_name DGCRN --seed 2023 --bs 64
```

