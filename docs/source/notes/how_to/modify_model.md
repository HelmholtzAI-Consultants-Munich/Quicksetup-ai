# How to set up a different model
In this Tutorial, we show how one can quickly edit the template to try a new architecture. In this example, we create a simple CNN architecture to classify MNIST images.
## Define the new architecture
In `src/ml_pipeline_template/models/components`, we create a new file called `simple_cnn.py`. For simplicity, we adapt the code from [pytorch documentation](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html) to create a CNN.

```
import torch.nn as nn
import torch.nn.functional as F
import torch


class SimpleCNN(nn.Module):
    def __init__(
        self,
        input_chans: int = 1,
        conv1_out_chans: int = 6,
        conv1_kernel_size: int = 5,

        conv2_out_chans: int = 16,
        conv2_kernel_size: int = 5,

        max_pool_size: int = 2,

        lin_input_size: int = 16 * 4 * 4,
        output_size: int = 10,
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(input_chans, conv1_out_chans, conv1_kernel_size)
        self.pool = nn.MaxPool2d(max_pool_size, max_pool_size)
        self.conv2 = nn.Conv2d(conv1_out_chans, conv2_out_chans, conv2_kernel_size)
        self.fc = nn.Linear(lin_input_size, output_size)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = self.fc(x)
        return x
```
## Create the configuration file
In `configs/model/`, we create a new file called `mnist_cnn.yaml` with the content of `mnist.yaml`. Then, we edit the `net` part by setting `_target_` to the new model, and providing the parameters we want for training. Here's the content of the new file.

```
_target_: ml_pipeline_template.models.mnist_module.MNISTLitModule
lr: 0.001
weight_decay: 0.0005

net:
  _target_: ml_pipeline_template.models.components.simple_cnn.SimpleCNN

  input_chans: 1
  conv1_out_chans: 6
  conv1_kernel_size: 5

  conv2_out_chans: 16
  conv2_kernel_size: 5

  max_pool_size: 2

  lin_input_size: 256
  output_size: 10
```
> For sake of simplicity, we keep on using MNISTLitModule since we do not need to alter the training process.

## Edit the main train/test configurations 
The final step is to configure `configs/train.yaml` and `configs/test.yaml`.

First, we edit the training configuration. Under `defaults`, we set `model` to `mnist_cnn.yaml` to make it use the new model. After training is completed, we edit the test configuration: We also set `model` to `mnist_cnn.yaml` under `defaults`. Then, we provide the `ckpt_path` to the model we want to test.

Congratulations! You can now use custom model architectures in the template.
