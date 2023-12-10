# DiCLS: A Deep Fusion Cross Modality Neural Network For Plant Disease Classification

# Introduction

This is a homework of Computer Vision in SYSU(Sun Yat-sen University).

This work investigated a lot on the dynamic mechanism of cross-modal aligmnet, but actually gained a little insights.

Related paper: [this page](./paper/paper.pdf).

# Quickstart

To run the code, please firstly clone the repo:

```
git clone https://github.com/ThreebodyDarkforest/DiCLS.git
```

Then, download the dataset from [this link](https://pan.baidu.com/s/1N10URTnXCaWbBWlRLISAWg#list/path=%2Fplant_dataset), unzip and place it to `data`.

Use the following command to run the codes:

```
python main.py --path /path/to/your/data --config /path/to/your/config
```

Note that the `path` is where you place the dataset(stop at folder name, like `data/plant_dataset`), and `config` is the settings of the whole model. We provided a sample in `config/config.yaml`.

To test your trained model, run the following command:

```
python main.py --test --path /path/to/your/data --weight /path/to/your/weight --config /path/to/your/config
```

# Acknowledgement

Firstly, thanks to my teacher in Computer Vision. I'm sincerely glad to listen to your lectures.

The same thanks to all the supports from my peer students and friends. It's nice to work with all of you.