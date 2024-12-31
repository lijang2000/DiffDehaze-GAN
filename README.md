# Extreme lighting Robust Neural Radiance Fields for Novel View synthesis
# Dataset
We use datasets in our experiments:

(1)LOM:https://drive.google.com/file/d/1orgKEGApjwCm6G8xaupwHKxMbT2s9IAG/view

(2)LLNeRF:https://drive.google.com/drive/folders/1h-u8DkvuaIvcHZihYIWcqwpURiM32_u3

(3)DarkGaussian:https://pan.baidu.com/share/init?surl=xmZqYEJ5ZMkdldPS9_MgiQ&pwd=jf48
## Dataset format

    datasets      
    │─── datasets-name
        │─── train
            │─── clear
                │─── 1.jpg
            │─── haze
        │─── test
            │─── clear
            │─── haze


# Environment
Python 3.10 and PyTorch 2.2.2 installed, and is configured with CUDA 12.6 to support GPU acceleration. A single NVIDIA RTX 4090 graphics card is utilized in the experiment.
<br/>
## Enviroment setup:
```
$ pip3 install -r requirements.txt
```



<br/>

# Running
python Train.py
