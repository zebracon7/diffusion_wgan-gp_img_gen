# Команда PuddleTech Кейс №2
**Файл с весами model-88.pt необходимо скачать по [ссылке](https://drive.google.com/file/d/1AaOvAYpjl3-FsEQ2WD_CtHW74REYTVTZ/view?usp=sharing) с Google Drive и разместить в папку results**

В данной работе реализованно 2 подхода к генерации ладоней: 
- Диффузионная модель на основании denoising-diffusion-pytorch
- WGAN-GP

На WGAN-GP заострять внимание не будем, тк она показала результат хуже. Описание по данной модели и файлы для запуска в папке wgan-gp

## Структура проекта
```
├── src
│   ├── __init__.py
│   ├── image_generation.py
│   ├── keypoint_detection.py
│   ├── model.py
│   ├── model_setup.py
│   ├── train.py
│   ├── trainer.py
│   └── transformations.py
├── data
│   └── processed_images
│       └── (dataset images)
├── results
│   └── model-88.pt
├── main.py
├── README.md
├── output.gif
├── requirements.txt
└── wgan-gp
```
## Запуск генерации изображений из репозитория
1. Установите зависимости pip install -r requirements.txt
2. Необходимо убедиться, что в папке results есть файл с [весами](https://drive.google.com/file/d/1AaOvAYpjl3-FsEQ2WD_CtHW74REYTVTZ/view?usp=sharing) model-88.pt.
3. Из склонированного репозитория из корня запускаем файл main.py с указанием папки (для сохранения сгенерированных изображений) и количества изображений, например: 
```
python main.py dirname 40
```
В результате, в указанном каталоге создаются изображения. Пример склеенных изображений в одну гифку: 

![пример генерации](output.gif)

## Запуск генерации изображений из Google Colab (все работы проводились в нем)
1. Необходимо создать папку в Google Drive под названием Diffusion_Model. Данная папка должна состоять из папок:
 - results (в нее положить файл с [весами](https://drive.google.com/file/d/1AaOvAYpjl3-FsEQ2WD_CtHW74REYTVTZ/view?usp=sharing))
 - в папке results - должна быть папка generated 
3. После чего запустить данный ноутбук в Google Colab: [ссылка](https://colab.research.google.com/drive/1l94Ig_zMgHkW_kxD5Iz8yRBSjNMVJdkg?usp=sharing) - для генерации изображений все соотвествующие ячейки 
4. Изображения сохранены в results/palm_variations

## Запуск обучения модели из репозитория
В папке data/processed_images/ должен находиться датасет. Наш тренировочный датасет находится по [ссылке](https://drive.google.com/drive/folders/1Iu5WQsy9tmPNHJarNQj0rbwmF05uUum6?usp=sharing) (состоит из изображений 128х128)

1. Запустите из корня проекта файл src/train.py:
```
python src/train.py  
```
2. Веса (файлы .pt) и промежуточные изображения для наблюдения и анализа динамики по качеству гиперпараметров сохраняются в папке results

## Запуск обучения модели из Google Colab
1. Содержание Google Drive должно быть аналогично п.1 из "Запуск генерации изображений из Google Colab"
2. Запустите соответствующие ячейки
3. Веса (файлы .pt) и промежуточные изображения для наблюдения и анализа динамики по качеству гиперпараметров сохраняются в папке results
