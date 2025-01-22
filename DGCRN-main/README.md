# The LargeST Benchmark Dataset

This is the official repository for the NeurIPS 2023 DB Track paper [LargeST: A Benchmark Dataset for Large-Scale Traffic Forecasting](https://arxiv.org/abs/2306.08259). The DGCRN model is integrated within LargeST, and we have incorporated the SGL module into this framework for experimentation.

## 1. Experiments Running

To run the DGCRN baseline, you may execute the Python file in the terminal:
```
python experiments/DGCRN/main.py --device cuda:2 --dataset PEMSD8 --years 2019 --model_name DGCRN --seed 2023 --bs 64
```


## 2. Citation
If you find our work useful in your research, please cite:
```
@inproceedings{liu2023largest,
  title={LargeST: A Benchmark Dataset for Large-Scale Traffic Forecasting},
  author={Liu, Xu and Xia, Yutong and Liang, Yuxuan and Hu, Junfeng and Wang, Yiwei and Bai, Lei and Huang, Chao and Liu, Zhenguang and Hooi, Bryan and Zimmermann, Roger},
  booktitle={Advances in Neural Information Processing Systems},
  year={2023}
}

```
