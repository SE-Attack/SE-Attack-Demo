# QT-Attack: A Simple and Effective Query- & Transfer-Based Paradigm for Black-Box Hard-Label Attack

<center> 
<img src='pic/flowsheet.png' width='800px'>
</center>


## Environment
See `requirements.txt`, some key dependencies are:

* python==3.8
* torch==1.11.0 


## Perform attacks

### Dataset

### Classifiers

```bash
# QT-Attack against the target model under the hard-label attack setting
bash run.sh 

# QT-Attack against Goolge Cloud Vision API
python attack_gcv.py
```

### Google cloud vision API
[*gcv_images.zip*](https://github.com/CSIPlab/BASES/raw/main/imagenet1000.zip) contains randomly selected images and responses from GCV

```
python gcv_attack.py
```



