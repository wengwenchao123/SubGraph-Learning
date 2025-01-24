## STAEformer: Spatio-Temporal Adaptive Embedding Transformer

This is a PyTorch implementation of **STAEformer: Spatio-Temporal Adaptive Embedding Transformer**. We have embedded SGL into STAEformer. Please follow the relevant guidelines to run the code.


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
