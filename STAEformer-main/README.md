## STAEformer: Spatio-Temporal Adaptive Embedding Transformer

#### H. Liu*, Z. Dong*, R. Jiang#, J. Deng, J. Deng, Q. Chen, X. Song#, "Spatio-Temporal Adaptive Embedding Makes Vanilla Transformer SOTA for Traffic Forecasting", Proc. of 32nd ACM International Conference on Information and Knowledge Management (CIKM), 2023. (*Equal Contribution, #Corresponding Author)

![model_arch](https://github.com/XDZhelheim/STAEformer/assets/57553691/f0620d5b-2b7f-47bc-bf76-5fccf48fae35)

#### Citation
```
@inproceedings{liu2023spatio,
  title={Spatio-temporal adaptive embedding makes vanilla transformer sota for traffic forecasting},
  author={Liu, Hangchen and Dong, Zheng and Jiang, Renhe and Deng, Jiewen and Deng, Jinliang and Chen, Quanjun and Song, Xuan},
  booktitle={Proceedings of the 32nd ACM International Conference on Information and Knowledge Management},
  pages={4125--4129},
  year={2023}
}
```


#### Required Packages

```
pytorch>=1.11
numpy
pandas
matplotlib
pyyaml
pickle
torchinfo
```

#### Training Commands

```bash
cd model/
python train.py -d <dataset> -g <gpu_id>
```

`<dataset>`:
- METRLA
- PEMSBAY
- PEMS03
- PEMS04
- PEMS07
- PEMS08
