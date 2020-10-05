# graph-convolutions
Relational Graph Convolutional Networks (R-GCNs) with PyTorch

---

Here are some experimentations on R-GCNs, based on the paper 
*Modeling Relational Data with Graph Convolutional Networks* from [[Schlichtkrull *et al.*]](https://arxiv.org/pdf/1703.06103.pdf).

### Requirements

You'll need PyTorch, numpy, scipy.


### Data

We use the [FB15K-237](https://www.microsoft.com/en-us/download/details.aspx?id=52312) dataset, from [[Toutanova et al. EMNLP 2015]](http://dx.doi.org/10.18653/v1/D15-1174).
It can be downloaded with:
```
wget https://download.microsoft.com/download/8/7/0/8700516A-AB3D-4850-B4BB-805C515AECE1/FB15K-237.2.zip
unzip FB15K-237.2.zip
```
Or you can use directly:
```
import fb15k
train, test = fb15k.load("train", "test", download_if_absent=True)
```

