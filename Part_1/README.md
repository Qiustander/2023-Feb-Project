# Part-1 Missing Data

## Overall 
Tensorflow2 implementation of 4 papers:
1. [Diffusion-based Time Series Imputation and Forecasting with Structured State Space Models (SSSD)](https://arxiv.org/abs/2208.09399)
2. [CSDI: Conditional Score-based Diffusion Models for Probabilistic Time Series Imputation](https://arxiv.org/abs/2107.03502)
3. [Structured State Space for Sequence Modeling (S4)](https://arxiv.org/abs/2111.00396)
4. [It's Raw! Audio Generation with State-Space Models (Sashimi)](https://arxiv.org/abs/2202.09729).

SSSD is based on [this GitHub repository](https://github.com/AI4HealthUOL/SSSD).

CSDI is based on [this GitHub repository](https://github.com/ermongroup/CSDI).

S4 is based on the beautiful Annotated S4 [blog post](https://srush.github.io/annotated-s4/)
and JAX-based [library](https://github.com/srush/annotated-s4/) by [@srush](https://github.com/srush) and 
[@siddk](https://github.com/siddk).

Sashimi is based on [this GitHub repository](https://github.com/HazyResearch/state-spaces/tree/main/sashimi).

## Requirement

Please install the packages in requirements.txt
Requires Python 3.9+.

## Experiments 

### training for the Mujoco dataset
```shell
python train.py
```
Can use different methods: CSDIS4, CSDI, SSSDS4, SSSSDSA

### Sampling results
`Inference.py` is for SSSD & `Inference_CDSI.py` is for SSSD.

### Visualize results
`Visualize.py` is for visualization after executing `Inference.py`