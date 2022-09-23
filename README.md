# Adversarial Reprogramming Revisited

The code in this repository can be used to run the experiments from our paper:

[**Adversarial Reprogramming Revisited**](https://arxiv.org/abs/2206.03466) <br>
*Matthias Englert, Ranko Lazic*

## Requirements

To install required python packages run:

```
pip install -r requirements.txt
```

## Usage

The program in adversarial_reprogramming.py can be used to find adversarial programs for random networks; repurposing them to work on different datasets such as mnist.

The following command line options are supported:

```
    --network: The network architecture to use.
    --dataset: The dataset to use. Defaults to mnist.
    --input_value_range: A value between 0 and 1 indicating how the pixel values of the input are scaled.
                         If 0, the input is ignored and only the adveserial program is fed into core_model.
                         If 1, in those pixels that contain the input image, there is no adveserial program used.
                         Defaults to 1.
    --image_size: The size of the image relative to the input size expected by the network. Defaults to 1.
    --epochs: The number of epochs to use to find an adversarial program. Defaults to 20.
    --lr: The learning rate to use for training. Defaults to 0.01.
    --batch_size: The batch size to use. Defaults to 50.
```

For example, the program can be called as:

```
python adversarial_reprogramming.py --network ResNet50V2 --dataset fashion_mnist --input_value_range 0.05 --epochs 5
```        

Large batch sizes require a certain amount of memory. If the available hardware does not provide enough memory, it may be necessary to reduce the batch size.
