**Data Efficient Contrastive Learning in Histopathology using Active Sampling**

https://arxiv.org/pdf/2303.16247

Deep learning based diagnostics systems can provide accurate and robust quantitative analysis in digital pathology. These algorithms require large amounts of annotated training data which is impractical in pathology due to the high resolution of histopathological images. Hence, self-supervised methods have been proposed to learn features using ad-hoc pretext tasks. The self-supervised training process uses a large unlabeled dataset which makes the learning process time consuming. In this work, we propose a new method for actively sampling informative members from the training set using a small proxy network, decreasing sample requirement by 93\% and training time by 62\% while maintaining the same performance of the traditional self-supervised learning method.


<img src="https://github.com/Reasat/data_efficient_cl/assets/15989033/25edabd1-2ec1-4106-a3d8-93161b8dc7cf" width="700" height="600">

**Usage**

The training and evaluation function are present in the `simclr_kather_active_sampling.py` script. The dataset can be downloaded from https://zenodo.org/records/1214456
