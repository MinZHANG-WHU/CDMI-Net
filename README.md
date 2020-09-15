# CDMI-Net

This repository contains code, network definitions and pre-trained models for a novel deep neural network (CDMI-Net). The CDMI-Net that combines change detection and multiple instance learning (MIL) is proposed for landslide mapping, see figure 1.

The implementation uses the [Pytorch](https://pytorch.org/) framework.

![](/img/Proposed_method.png)
<center>Figure 1. The workflow of the proposed method..</center>


## Motivation
In a large geographic area, due to the lack of enough landslides (non-landslide is the vast majority), the landslide detection is often considered as a low-likelihood pattern detection rather than a binary classification problem. Therefore, using scene-level change detection is a good solution to filter out most unchanged scenes. In this work, we use the MIL framework to enable CNN to automatically learn the deep features of landslides from scene-level samples, thereby reducing the need for pixel-level samples. A two-stream U-Net with shared weight is designed as a deep feature extractor (Change detection technique), which can help remove ground objects that have similar characteristics to landslides but do not change over time. 

The method is proposed to alleviate two problems in change detection:  1) insufficient pixel-level labeled samples, and 2) low-likelihood of changes. In addition, the proposed method can also be used to detect changes in other ground objects.

## Content

### Datasets

Our data sets are currently not available to the public. Since only scene-level annotations are required, users can quickly generate the training data sets. To train the CDMI-Net, each image pair (i.e. bag) is labeled as “positive bag” (P) or “negative bag” (N).The file organization is as follows: 

```
data_dir        
   ├── P/                          # positive bags                 
   │   ├── bag1_T1.tif             # first period image
   │   ├── bag1_T2.tif             # second period image
   │   ├── bag2_T1.tif             
   │   ├── bag2_T2.tif             
   │   ...  
   │   ├── bagM_T1.tif             
   │   ├── bagM_T2.tif             
   |   
   ├── N/                          # negative bags         
   │   ├── bag1_T1.tif             # first period image
   │   ├── bag1_T2.tif             # second period image
   │   ├── bag2_T1.tif            
   │   ├── bag2_T2.tif             
   │   ...  
   │   ├── bagM_T1.tif             
   │   ├── bagM_T2.tif               
   |
   ├── train.txt
   ├── test.txt
```

### How to start

##### 1. Environments

This code was developed and tested with Python 3.6.2 and PyTorch 1.0.1.

```
**Install dependencies**
numpy                    1.18.1
GDAL                     2.4.1
opencv-python            4.1.1.26
torchvision              0.4.1
tqdm                     4.41.0
Pillow                   6.2.1
```

##### 2. Training

To train the CDMI-Net on the training set and  test its accuracy on the test set, run the following commands:

```
python mil_train.py --data_dir DATA_DIR --weight_dir WEIGHT_DIR
```

* DATA_DIR: the absolute path of the training data set.
* WEIGHT_DIR: the absolute path to the saved check point.

##### 3. Testing

 Using your own trained CDMI-Net model, or download our [pre-trained CDMI-Net model](https://drive.google.com/file/d/12qBG5QztBB1TXGg25jaoJI1IKfuPdYLX/view?usp=sharing). To Test the accuracy of CDMI-Net on large geographic area, run the following commands:

```
python mil_infer.py --t1 T1_IMAGE_PATH --t2 T2_IMAGE_PATH --weight CHECK_POINT_PATH--save-dir OUTPUT_PATH --gt GT_PATH 
```
* T1_IMAGE_PATH: the first period image path (t1 must be 3 bands).
* T2_IMAGE_PATH: the second period image path (Its image size must be the same of t1).
* CHECK_POINT_PATH: the pre-trained CDMI-Net model path.
* OUTPUT_PATH: the output path for the predicted results.
* GT_PATH: the ground truth image path (0 and 255 represent unchanged and changed respectively).

Finally, pixel score map (pixel_score.tif), binary image (pixel_bm.tif), binary image with FLSE (pixel_flse.tif), and scene level result (scene.tif) will be generated under the "OUTPUT_PATH" folder.

## References
If you use this work for your projects, please take the time to cite our [paper](https://doi.org/10.1109/LGRS.2020.3007183).

```
@ARTICLE{9142246,
  AUTHOR={Zhang, Min and Shi, Wenzhong and Chen, Shanxiong and Zhan, Zhao and Shi. Zhicheng},
  JOURNAL={IEEE Geoscience and Remote Sensing Letters}, 
  TITLE={Deep Multiple Instance Learning for Landslide Mapping}, 
  YEAR={2020},
  VOLUME={},
  NUMBER={},
  PAGES={1-5},
  DOI={10.1109/LGRS.2020.3007183},
  ISSN={1558-0571},
  MONTH={},
  }
```

## License
Code and models are released under the GPLv3 license for non-commercial and research purposes **only**. For commercial purposes, please contact the authors.