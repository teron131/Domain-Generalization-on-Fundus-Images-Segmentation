# Domain Generalization on Fundus Images Segmentation
***Colaboration with [UrMBCMRabbont](https://github.com/UrMBCMRabbont)***

*The Final Project 2 of HKUST ELEC4010N - Artificial Intelligence for Medical Image Analysis*

Implementing domain generalization of multi-class segmentation on fundus images segmentation dataset by Fourier Augmented Co-Teacher (FACT) model and U-Net.

For more high-level details, read the Project 2 part of the [presentation slides](./Presentation.pdf) and the [report](./Report.pdf).

Results:

<table>
  <tr>
    <th>Train</th>
    <th>Test</th>
    <th>Model</th>
    <th>Mean Test Dice</th>
    <th>OC Test ASD</th>
    <th>OD Test ASD</th>
  </tr>
  <tr>
    <td rowspan="2">123</td>
    <td rowspan="2">4</td>
    <td>Baseline</td>
    <td>0.5781</td>
    <td>36.9649</td>
    <td>27.7053</td>
  </tr>
  <tr>
    <td>FACT</td>
    <td><b>0.8730</b></td>
    <td><b>7.4794</b></td>
    <td><b>1.7167</b></td>
  </tr>
  <tr>
    <td rowspan="2">124</td>
    <td rowspan="2">3</td>
    <td>Baseline</td>
    <td>0.6057</td>
    <td>35.9788</td>
    <td>24.8685</td>
  </tr>
  <tr>
    <td>FACT</td>
    <td><b>0.9039</b></td>
    <td><b>6.2443</b></td>
    <td><b>0.6492</b></td>
  </tr>
  <tr>
    <td rowspan="2">134</td>
    <td rowspan="2">2</td>
    <td>Baseline</td>
    <td>0.6988</td>
    <td>24.0777</td>
    <td>15.9232</td>
  </tr>
  <tr>
    <td>FACT</td>
    <td><b>0.8527</b></td>
    <td><b>8.2105</b></td>
    <td><b>1.4624</b></td>
  </tr>
  <tr>
    <td rowspan="2">234</td>
    <td rowspan="2">1</td>
    <td>Baseline</td>
    <td>0.6376</td>
    <td>30.5020</td>
    <td>21.3653</td>
  </tr>
  <tr>
    <td>FACT</td>
    <td><b>0.8996</b></td>
    <td><b>5.3635</b></td>
    <td><b>1.2530</b></td>
  </tr>
</table>

## Prerequisites
Download and unzip data from the link [Fundus dataset](https://drive.google.com/u/0/uc?id=1p33nsWQaiZMAgsruDoJLyatoq5XAH-TH&export=download).

Place the `Fundus` folder (not `Fundus-doFE`)  into the main directory. If running in Colab, change the paths accordingly and run the following commands in the notebook:

```python
from google.colab import drive
drive.mount('/content/gdrive')
%cd "/content/gdrive/MyDrive/Colab Notebooks/.../your_project_folder"
!unzip "/content/gdrive/MyDrive/Colab Notebooks/.../your_project_folder/Fundus-doFE.zip" -d "/content/"
```

Install the additional libraries by:

```python
!pip install segmentation-models-pytorch
!git clone https://github.com/deepmind/surface-distance.git
!pip install surface-distance/
```

For the requirements of `segmentation-models-pytorch`, install the packages by `pip install -r requirements.txt` or in the notebook:

```python
!pip install torchvision>=0.5.0
!pip install pretrainedmodels==0.7.4
!pip install efficientnet-pytorch==0.7.1
!pip install timm==0.6.13
!pip install tqdm
!pip install pillow
```

These parts are included in the first two code cells in the notebook.

## Notebook Outline
0. For Colab
1. Import
2. Fundus Dataset
3. Segmentation Baseline
    1. U-Net
    2. Average Surface Distance (ASD)
4. Baseline Experiment
    1. Training
    2. Results
    3. Evaluation
5. FACT
    1. Utilities
    2. Fourier Augmentation
    3. Mean Teacher Model
    4. Training
    5. Results
    6. Evaluation

## Reference
- Wang, S., Yu, L., Li, K., Yang, X., Fu, C.-W., Heng, P.-A. (2020). DoFE: Domain-oriented Feature Embedding for
Generalizable Fundus Image Segmentation on Unseen Datasets. IEEE Transactions on Medical Imaging.
(https://github.com/emma-sjwang/Dofe)
- Xu, Q., Zhang, R., Zhang, Y., Wang, Y., Tian, Q. (2021). A Fourier-Based Framework for Domain Generalization. In
Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR).
(https://github.com/MediaBrain-SJTU/FACT)
- Laine, S., & Aila, T. (2017). Temporal Ensembling for Semi-Supervised Learning. International Conference on
Learning Representations (ICLR). *arXiv:1610.02242*
- Kim, T., Oh, J., Kim, N., Cho, S., & Yun, S. (2021). Comparing Kullback-Leibler Divergence and Mean
Squared Error Loss in Knowledge Distillation. Proceedings of International Joint Conference on Artificial
Intelligence (IJCAI). *arXiv:2105.08919*
