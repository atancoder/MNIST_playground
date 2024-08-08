# MNIST
Playing around with the MNIST dataset using ML

## CNN Model
Create a CNN model to classify handwritten images of digits (MNIST dataset)

1. Train the model
Currently 10 epochs, which takes 1 min to train
```
py main.py train_model
```

2. Test the model
```
py main.py test_model
```

Accuracy is 98%


## Fine Tuning ViT
Fine tune [ViT](https://huggingface.co/google/vit-base-patch16-224) on the MNIST dataset

```
py main.py fine_tune  --saved_model atancoder/ViT_fine_tuned_mnist
```
Accuracy is 99.6%


## Generative Eneergy Model
Create a simple energy model that can generate images

Linear model works better than CNN. Images generated not as good as VAE.
Had to subset data to a particular class label to get it working decently.

```
cd energy_model
py main.py train_model
py main.py gen_images

```

For linear model
- LR: 1e-5
- LD step size: 1e-3

For CNN model
- LR: 1e-4
- LD step size: 1e-1


## VAE
Works great! Trained for 50 epochs

```
cd VAE
py main.py train_model
py main.py gen_images
```