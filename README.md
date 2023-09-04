# Launching the app

```commandline
conda create -n bg-rem -f environment.yml
conda activete bg-rem
uvicorn app:app
http://127.0.0.1:5003/
```


# Application implementation to remove image background

## 1. Existing approaches and architecture choices

Background removal is one of the tasks of computer vision.<br>
In scientific articles, it can be found by the keywords "Foreground Segmentation", "Background Subtraction" or "Image Matting".<br>
One of the first approaches was described back in the 90s [of the last century](#source_1) and was based on the representation of an image as a graph consisting not of pixels, but of nodes, each of which belonged either to the background or to an object.<br>

Later, other methods of background segmentation appeared, but they were all united by the fact that the border of the object was too rough and missed part of the pixels belonging to the background, or captured part of the object. <br>
To clarify the boundaries of the object, they began to use an approach called "matting". The method of border refinement is described in detail, for example, in [article 2013](#source_2) of the year. <br>
In general, there is a frame (trimap), in which each pixel is represented as a combination of background and object, then a sliding window, as when convolving in CNN, passes through the frame and, depending on the value obtained during this operation, the pixel is classified either as a background or as an object.<br>
The image shows how the "border refinement" removes artifacts after the first pass of the segmentation algorithm:


Since after the success of AlexNet in 2012, convolutional neural networks received well-deserved attention and in the 10s, many architectures appeared to solve a variety of computer vision problems, the problem of separating an object from the background received a qualitatively new solution.<br>
So in 2017, a study by Adobe was born, called ["Deep Image Matting"](#source_3), in which the very approach of "clarifying boundaries" was implemented based on deep learning methods. <br>
The new matting used a neural network consisting of 2 stages:
1. Encode-Decoder stage was used to get groundtrooth (and by itself gave better results than the previous approach)
2. The second stage is needed to clarify the results.


Among the latest works, the work of [2021](#source_4) should be noted. <br>
The authors of the article used a recurrent neural network, which allowed them to remove the background not only from images, but also from videos. <br>
Frames are compressed when fed into the neural network, and after passing through complex recurrent layers, the groundtrooth is restored to the original resolution and refined. This allows you to process high-resolution video using a minimum of computing power. <br>
The result (according to the authors) is 4K video processing (70 FPS) in real time on the GPU

To implement an application that removes the background from the picture, I will use this model. <br>

## 2. Datasets for the task of allocating won:


* [AM-2k](https://github.com/JizhiziLi/GFM#am-2k ) - two thousand images of animals in high resolution
* [AIM-500](https://github.com/JizhiziLi/AIM#aim-500 ) - a dataset with photos taken in natural conditions and marked up manually. It consists of 500 images.
* [P3M-10k](https://github.com/JizhiziLi/P3M#ppt-setting-and-p3m-10k-dataset ) - a dataset of 10 thousand photos of people obtained in vivo.
* [PPM-100](https://github.com/ZHKKKe/PPM#download ) - 100 portraits obtained from Flickr.
* [Distinctions-646](https://github.com/vietnamican/HAttMatting ) - Dataset consisting of 646 images marked up manually
* [AISegment.com - Matting Human Datasets](https://www.kaggle.com/datasets/laurentmih/aisegmentcom-matting-human-datasets?resource=download ) - a dataset of more than 34 thousand images of people.
* [BG-20k](https://github.com/JizhiziLi/GFM#bg-20k ) - 20 thousand background images to create synthetic data

## 3. Working with data

Since the authors of the chosen model managed not only to collect datasets that are not publicly available (Distinctions-646 and Adobe Image Matting), but also to train their network with impressive computing power (`48 CPU cores, 300G CPU memory, and 4 Nvidia V100 32G GPUs`), I consider it appropriate to use their weights for solutions to the problem. <br>

If a demonstration of working with data is required, I have a combined dataset for segmentation, localization and classification of frames taken from a drone. Such surveys are used to track and prevent the spread of forest fires.

You can read about how I combined the data into one dataset here: [description](https://github.com/german-leontiev/uwf_data/blob/main/notebook.ipynb )<br>
The repository with the code for working with the dataset is located at the link: [uwf_data](https://github.com/german-leontiev/uwf_data )

Among other things, an algorithm for dividing data into test and validation samples is implemented there

## 4. Model for the application

For interest, you need to write a simple script. At the same time, take into account that there will be no CUDA on the server:


```python
import torch
from PIL import Image
import torchvision.transforms.functional as TF

model = torch.hub.load("PeterL1n/RobustVideoMatting", "mobilenetv3")
model = model.eval().cpu()


def predict(pth_, model):
    bgr = torch.tensor([.47, 1, .6]).view(3, 1, 1).cuda()  # Green background.
    rec = [None] * 4                                       # Initial recurrent states.
    downsample_ratio = 0.25
    image = Image.open(pth_)
    x = TF.to_tensor(image)
    x.unsqueeze_(0)
    fgr, pha, *rec = model(x.cpu(), *rec, downsample_ratio)
    np_arr_alph =  pha.cpu().detach().numpy()[0,0,:,:]
    alpha = Image.fromarray((np_arr_alph*255).astype('uint8'))
    image.putalpha(alpha)
    return image
```



##5. Web interface
The code I used to write the interface is in [on github](https://github.com/german-leontiev/bg-removal )

## 6. List of sources:
<a id='source_1'></a>
1. [J. Shi, S. Belongie, T. Leung and J. Malik, "Image and video segmentation: the normalized cut framework," Proceedings 1998 International Conference on Image Processing. ICIP98 (Cat. No.98CB36269), 1998, pp. 943-947 vol.1, doi: 10.1109/ICIP.1998.723676.](https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.128.5631&rep=rep1&type=pdf)
<a id='source_2'></a>
2. [Yu, Xintong & Liu, Xiaohan & Yisong, Chen. (2014). Foreground segmentation based on multi-resolution and matting.](https://arxiv.org/ftp/arxiv/papers/1402/1402.2013.pdf)
<a id="source_3"></a>
3. [Ning Xu, Brian Price, Scott Cohen, Thomas Huang. "Deep Image Matting" (2017)](https://arxiv.org/pdf/1703.03872.pdf)
<a id="source_4"></a>
4. [Shanchuan Lin and Linjie Yang and Imran Saleemi and Soumyadip Sengupta "Robust High-Resolution Video Matting with Temporal Guidance" (2021)](https://arxiv.org/pdf/2108.11515.pdf)
