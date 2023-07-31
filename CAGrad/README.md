# MoCo : Provably Convergent Stochastic Multi-objective Optimization

**This repository is adapted from the code base of [Conflict-Averse Gradient Descent for Multitask Learning (CAGrad)](https://github.com/Cranial-XIX/CAGrad)
We use this code base to implement MoCo, and comapre with the relevant baselines. Following are the original instructions provided by the authors of aforementioned work, and additional inforamtion to run MoCo.**


## Image-to-Image Prediction
The supervised multitask learning experiments are conducted on NYU-v2 and CityScapes datasets. We follow the setup from [MTAN](https://github.com/lorenmt/mtan). The datasets could be downloaded from [NYU-v2](https://www.dropbox.com/sh/86nssgwm6hm3vkb/AACrnUQ4GxpdrBbLjb6n-mWNa?dl=0) and [CityScapes](https://www.dropbox.com/sh/gaw6vh6qusoyms6/AADwWi0Tp3E3M4B2xzeGlsEna?dl=0). After the datasets are downloaded, please follow the respective run.sh script in each folder. In particular, modify the dataroot variable to the downloaded dataset. 

**To run the MoCo implementaion for each dataset, run run_moco.sh script in respective folders.**

## Citations
If you find our work interesting or the repo useful, please consider citing [this paper](https://arxiv.org/pdf/2110.14048.pdf):
```
@article{liu2021conflict,
  title={Conflict-Averse Gradient Descent for Multi-task Learning},
  author={Liu, Bo and Liu, Xingchao and Jin, Xiaojie and Stone, Peter and Liu, Qiang},
  journal={Advances in Neural Information Processing Systems},
  volume={34},
  year={2021}
}
```
