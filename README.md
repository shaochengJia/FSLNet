# FSLNet
Pytorch implementation of FSLNet proposed in paper "Joint Learning of Frequency and Spatial Domains for Dense Image Prediction" by Shaocheng JIA and Wei YAO [Arxiv](https://arxiv.org/abs/2202.08991).

# Overview
![graphicalabstract](https://user-images.githubusercontent.com/48814384/199497474-e9a0e1be-bf02-4ae2-bf77-77e2cbcfc9aa.jpg)

# Requirements 
imageio             2.9.0   
importlib-metadata  4.8.1  
jupyter             1.0.0  
matplotlib          3.4.3  
notebook            6.4.3  
numpy               1.20.2  
opencv-python       4.5.3.56  
pandas              1.3.4  
Pillow              8.3.1  
scikit-image        0.18.3  
scikit-learn        1.0.2  
scipy               1.7.1  
tensorboardX        2.4  
torch               1.9.1  

# Quick start
Please refer to [test.ipynb](./test.ipynb) to quickly test the models.

# Evaluation and training
Please refer to [Monodepth2](https://github.com/nianticlabs/monodepth2) for detailed evaluation and training.

# Weights
Please find the weights trained on the [KITTI dataset](https://www.cvlibs.net/datasets/kitti/) in [weights](./weights) folder.

# Citation
@article{jia2022joint,  
   title={Joint Learning of Frequency and Spatial Domains for Dense Image Prediction},  
   author={Jia, Shaocheng and Yao, Wei},  
   journal={ISPRS Journal of Photogrammetry and Remote Sensing},  
   volume={-},  
   pages={-},  
   year={2022},  
   publisher={Elsevier}  
}  
or  
@article{jia2022joint,  
  title={Joint Learning of Frequency and Spatial Domains for Dense Predictions},  
  author={Jia, Shaocheng and Yao, Wei},  
  journal={arXiv preprint arXiv:2202.08991},  
  year={2022}  
}  

