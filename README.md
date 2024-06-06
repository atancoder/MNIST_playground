# MNIST
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
py main.py fine_tune
```
Accuracy is 99.6%


