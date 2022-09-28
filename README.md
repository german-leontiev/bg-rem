# Запуск приложения

```commandline
conda create -n bg-rem -f environment.yml
conda activete bg-rem
uvicorn app:app
http://127.0.0.1:5003/
```


# Реализация приложения для удаления фона изображения

## 1. Существующие подходы и выбор архитектуры

Удаление фона является одной из задач компьютерного зрения.<br>
В научных статьях её можно найти по ключевым словам "Foreground Segmentation", "Background Subtraction" или "Image Matting".<br>
Один из первых подходов был описан ещё в 90-х годах [прошлого века](#source_1) и основывался на представлении изображения, как графа, состоящего не из пикселей, а из узлов, каждый из которых относился либо к фону, либо к объекту.<br>

В последствии появились и другие методы сегментации фона, но всех их объединяло то, что граница объекта была слишком грубой и упускала часть пикселей, кринадлежащих фону, или захватывала часть от объекта. <br>
Для уточнения границы объекта стали использовать подход, который называется "matting". Подробно метод уточнения границы описан, например, в [статье 2013](#source_2) года. <br>
В общих чертах, выделяется обрамление (trimap), в рамках которого каждый пиксель представляется в качестве совокупности фона и объекта, затем скользящее окно, как при свёртке в CNN проходится по обрамлению и в зависимости от значения, полученного при этой операции, пиксель классифицируется либо как фон, либо как объект.<br>
На изображении видно, как "уточнение границы" убирает артефакты после первого прохода сегментационного алгоритма: 


Поскольку после успеха AlexNet в 2012, свёрточные нейросети получили заслуженное внимание и в 10-х годах появилось множество архитектур для решения самых разных задач компьютерного зрения, проблема отделения объекта от фона получила качественно новое решение.<br>
Так в 2017 году на свет появилось исследование компании Adobe, под названием ["Deep Image Matting"](#source_3), в которой тот самый подход "уточнения границ" был реализован на основании deep learning методов. <br>
Новый matting использовал нейросеть, состоящую из 2-х стадий:
1. Encode-Decoder стадия использовалась для получения groundtrooth (и сама по себе давала лучшие результаты, чем предыдущий подход)
2. Вторая стадия нужна для уточнения  результатов.


Среди последних работ следует отметить работу [2021 года](#source_4). <br>
Авторы статьи использовали рекуррентную нейронную сеть, что позволило им удалять фон не только из изображений, но и из видео. <br>
Кадры при подаче в нейронную сеть сжимается, а после прохождения сложных рекурентных слоёв, groundtrooth восстанавливается до исходного разрешения и уточняется. Это позволяет обрабатывать видео большого разрешения, используя минимум вычислительных мощностей. <br>
Результат (по заявлениям авторов) - обработка 4К видео (70 FPS) в режиме реального времени на GPU

Для реализации приложения, удаляющего фон с картинки, я буду использовать именно эту модель. <br>

## 2. Датасеты для задачи выделения вона:


* [AM-2k](https://github.com/JizhiziLi/GFM#am-2k) - две тысячи изображений животных в высоком разрешении
* [AIM-500](https://github.com/JizhiziLi/AIM#aim-500) - датасет с фотографиями, полученными в естественных условиях и размеченными вручную. Состоит из 500 изображений.
* [P3M-10k](https://github.com/JizhiziLi/P3M#ppt-setting-and-p3m-10k-dataset) - датасет из 10 тысяч фотографий людей, полученных в естественных условиях.
* [PPM-100](https://github.com/ZHKKKe/PPM#download) - 100 портретов, полученных из Flickr.
* [Distinctions-646](https://github.com/vietnamican/HAttMatting) - Датасет, состоящий из 646 изображений, размеченных вручную
* [AISegment.com - Matting Human Datasets](https://www.kaggle.com/datasets/laurentmih/aisegmentcom-matting-human-datasets?resource=download) - датасет из более чем 34 тысяч изображений людей.
* [BG-20k](https://github.com/JizhiziLi/GFM#bg-20k) - 20 тысяч фоновых изображений, для создания синтетических данных

## 3. Работа с данными

Поскольку авторам выбранной модели удалось не только собрать датасеты, не находящиеся в публичном доступе (Distinctions-646 и Adobe Image Matting), но и обучить свою сеть на внушительных вычислительных мощностях (`48 CPU cores, 300G CPU memory, and 4 Nvidia V100 32G GPUs`), я считаю целесообразным использовать их веса для решения задачи. <br>

Если требуется демонстрация работы с данными, у меня есть комбинированный датасет лесных для сегментации, локализации и классификации кадров, снятых с дрона. Такие съёмки используются для отслеживания и предупреждения распространения лесных пожаров.

Про то каким образом я объединял данные в один датасет можно почитать тут: [описание](https://github.com/german-leontiev/uwf_data/blob/main/notebook.ipynb)<br>
Репозиторий с кодом для работы с датасетом находится со ссылке: [uwf_data](https://github.com/german-leontiev/uwf_data)

Там в том числе реализован алгоритм разделения данных на тестовую и валидационную выборки

## 4. Модель для приложения

Для инференса нужно написать простой скрипт. При этом учтём, что на сервере не будет CUDA:


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



## 5. Веб-интерфейс
Готовый сервис находится [на моём сайте](https://bg-removal.german-leontiev.ml). *Скорость работы зависит от размера изображения.*<br>
Код, который я использовал для написания интерфейса находится в [на github](https://github.com/german-leontiev/bg-removal)

## 6. Список источников:
<a id='source_1'></a>
1. [J. Shi, S. Belongie, T. Leung and J. Malik, "Image and video segmentation: the normalized cut framework," Proceedings 1998 International Conference on Image Processing. ICIP98 (Cat. No.98CB36269), 1998, pp. 943-947 vol.1, doi: 10.1109/ICIP.1998.723676.](https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.128.5631&rep=rep1&type=pdf)
<a id='source_2'></a>
2. [Yu, Xintong & Liu, Xiaohan & Yisong, Chen. (2014). Foreground segmentation based on multi-resolution and matting.](https://arxiv.org/ftp/arxiv/papers/1402/1402.2013.pdf)
<a id="source_3"></a>
3. [Ning Xu, Brian Price, Scott Cohen, Thomas Huang. "Deep Image Matting" (2017)](https://arxiv.org/pdf/1703.03872.pdf)
<a id="source_4"></a>
4. [Shanchuan Lin and Linjie Yang and Imran Saleemi and Soumyadip Sengupta "Robust High-Resolution Video Matting with Temporal Guidance" (2021)](https://arxiv.org/pdf/2108.11515.pdf)
