# Extreme lighting Robust Neural Radiance Fields for Novel View synthesis
# Dataset
We use datasets in our experiments:

(1)RESIDE:[click here..](https://sites.google.com/view/reside-dehaze-datasets/reside-standard?authuser=0)

(2)BeDDE:[click here..](https://github.com/xiaofeng94/BeDDE-for-defogging)

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
