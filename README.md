# SEM-Attack: A Smart and Effective Method for  

<center> 
<img src='pic/flowsheet.pdf' width='800px'>
</center>


## Environment
See `requirements.txt`, some key dependencies are:

* python==3.8
* torch==1.11.0 


## Perform attacks

### Classifiers

```bash
# SEM-Attack under the decision-based attack setting
bash decison_attack.sh

# SEM-Attack under the score-based attack setting
bash score_attack.sh
```

### Google cloud vision API
[*gcv_images.zip*](https://github.com/CSIPlab/BASES/raw/main/imagenet1000.zip) contains randomly selected images and responses from GCV

```
python gcv_attack.py
```



