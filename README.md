#  Hyperparameter Analysis for Image Captioning

We perform a thorough sensitivity analysis on state-of-the-art image captioning approaches using two different architectures: CNN+LSTM and CNN+Transformer. Experiments were carried out using the Flickr8k dataset. The biggest takeaway from the experiments is that fine-tuning the CNN encoder outperforms the baseline and all other experiments carried out for both architectures. A detailed paper for this project is available here: https://github.com/aravindvarier/Image-Captioning-Pytorch/blob/master/Hyperparameter_Analysis_for_Image_Captioning.pdf

If you have any questions related to this, please reach out to us by creating an issue on this repository or through our emails listed in the paper.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites
1. Download the [Flickr8k Dataset](https://www.kaggle.com/shadabhussain/flickr8k) and place it under `dataset` folder of this directory.
2. Execute the following commands in this folder to set up the require virtual environment for running these experiments.

```
python3 -m venv proj_env
source proj_env/bin/activate
pip install -r requirements.txt
```

3. Generate the Vocab file:
```
python vocab_builder.py
```

## Running the experiments
Please execute the following commands in order to reproduce the results discussed in this paper. Please note that the results of the experiment is stored as csv files under `/experiments` folder and gets updated automatically once an experiment has been executed successfully.

### CNN + LSTM
There were a total of 3 experiments performed for this architecture.

1. Effect of larger CNN models on caption quality (ResNet18, ResNet50, and ResNet101):
```
python train.py --encoder-type resnet18 --experiment-name  resnet18_h512_bs64_ft0
python train.py --encoder-type resnet50 --experiment-name resnet50_h512_bs64_ft0
python train.py --encoder-type resnet101 --experiment-name resnet101_h512_bs64_ft0
```

2. Effect of finetuning on caption quality (ResNet18, ResNet50, and ResNet101):
```
python train.py --encoder-type resnet18 --experiment-name resnet18_h512_bs64_ft1 --fine-tune 1
python train.py --encoder-type resnet50 --experiment-name resnet50_h512_bs32_ft1 --fine-tune 1 --batch-size 32
python train.py --encoder-type resnet101 --experiment-name resnet101_h512_bs32_ft1 --fine-tune 1 --batch-size 32
```

3. Effect of varying LSTM units (keeping encoder fixed and varying decoder):

* Using ResNet18:
    ```
    python train.py --decoder-hidden-size 256 --encoder-type resnet18 --experiment-name resnet18_h256_bs64_ft0
    python train.py --decoder-hidden-size 512 --encoder-type resnet18 --experiment-name resnet18_h512_bs64_ft0
    python train.py --decoder-hidden-size 1024 --encoder-type resnet18 --experiment-name resnet18_h1024_bs64_ft0
    ```

* Using ResNet50:
  ```
  python train.py --decoder-hidden-size 256 --encoder-type resnet50 --experiment-name resnet50_h256_bs64_ft0
  python train.py --decoder-hidden-size 512 --encoder-type resnet50 --experiment-name resnet50_h512_bs64_ft0
  python train.py --decoder-hidden-size 1024 --encoder-type resnet50 --experiment-name resnet50_h1024_bs32_ft0 --batch-size 32 
  ```

* Using ResNet101:
  ```
  python train.py --decoder-hidden-size 256 --encoder-type resnet101 --experiment-name resnet101_h256_bs64_ft0
  python train.py --decoder-hidden-size 512 --encoder-type resnet101 --experiment-name resnet101_h512_bs64_ft0
  python train.py --decoder-hidden-size 1024 --encoder-type resnet101 --experiment-name resnet101_h1024_bs32_ft0 --batch-size 32 
  ```
  
### CNN + Transformer
There were a total of 3 experiments performed for this architecture.

1. Effect of larger CNN models on caption quality (ResNet18, ResNet50, and ResNet101):
```
python train.py --encoder-type resnet18 --decoder-type transformer --num-heads 1 --num-tf-layers 3  --experiment-name resnet18_bs64_ft0_l3_h1
python train.py --encoder-type resnet50 --decoder-type transformer --num-heads 1 --num-tf-layers 3  --experiment-name resnet18_bs64_ft0_l3_h1
python train.py --encoder-type resnet101 --decoder-type transformer --num-heads 1 --num-tf-layers 3  --experiment-name resnet18_bs64_ft0_l3_h1
```

2. Effect of finetuning on caption quality (ResNet18, ResNet50, and ResNet101):
```
python train.py --encoder-type resnet18 --decoder-type transformer --num-heads 1 --num-tf-layers 3  --fine-tune 1 --experiment-name resnet18_bs64_ft1_l3_h1
python train.py --encoder-type resnet50 --decoder-type transformer --num-heads 1 --num-tf-layers 3  --fine-tune 1 --experiment-name resnet18_bs64_ft1_l3_h1
python train.py --encoder-type resnet101 --decoder-type transformer --num-heads 1 --num-tf-layers 3 --fine-tune 1 --experiment-name resnet18_bs64_ft1_l3_h1
```

3. Effect of varying number of transformer layers and heads (keeping encoder fixed as ResNet18 and varying decoder):

* Using 1 Head:
    ```
    python train.py --encoder-type resnet18 --decoder-type transformer --num-heads 1 --num-tf-layers 3 --experiment-name resnet18_bs64_ft0_l3_h1
    python train.py --encoder-type resnet18 --decoder-type transformer --num-heads 1 --num-tf-layers 5 --experiment-name resnet18_bs64_ft0_l5_h1
    python train.py --encoder-type resnet18 --decoder-type transformer --num-heads 1 --num-tf-layers 7 --experiment-name resnet18_bs64_ft0_l7_h1
    ```

* Using 2 Head:
    ```
    python train.py --encoder-type resnet18 --decoder-type transformer --num-heads 2 --num-tf-layers 3 --experiment-name resnet18_bs64_ft0_l3_h2
    python train.py --encoder-type resnet18 --decoder-type transformer --num-heads 2 --num-tf-layers 5 --experiment-name resnet18_bs64_ft0_l5_h2
    python train.py --encoder-type resnet18 --decoder-type transformer --num-heads 2 --num-tf-layers 7 --experiment-name resnet18_bs64_ft0_l7_h2
    ```
* Using 3 Head:
    ```
    python train.py --encoder-type resnet18 --decoder-type transformer --num-heads 3 --num-tf-layers 3 --experiment-name resnet18_bs64_ft0_l3_h3
    python train.py --encoder-type resnet18 --decoder-type transformer --num-heads 3 --num-tf-layers 5 --experiment-name resnet18_bs64_ft0_l5_h3
    python train.py --encoder-type resnet18 --decoder-type transformer --num-heads 3 --num-tf-layers 7 --experiment-name resnet18_bs64_ft0_l7_h3
    ```


## Results Visualization Notebooks
1. CNN+LSTM: https://github.com/aravindvarier/Image-Captioning-Pytorch/blob/master/experiments/CNN%2BLSTM_Results.ipynb
2. CNN+Transformer: https://github.com/aravindvarier/Image-Captioning-Pytorch/blob/master/experiments/CNN%2BTransformer_Results.ipynb

## Built With

* [PyTorch](https://pytorch.org/) 
