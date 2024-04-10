# **Assignment 9**

---

1. Write a new network with the following architecture:
   - C1|C2|C3|C4|O (No MaxPooling, but 3 layers with a 3x3 kernel and stride of 2)
   - Total receptive field (RF) must be more than 44
   - One of the layers must use Depthwise Separable Convolution
   - One of the layers must use Dilated Convolution
   - Use Global Average Pooling (GAP) and optionally add a fully connected layer after GAP to target the desired number of classes
   - Apply the following augmentations using the albumentation library:
     - Horizontal flip
     - ShiftScaleRotate
     - CoarseDropout (max_holes = 1, max_height = 16, min_width = 16, min_holes = 1, min_height = 16, max_width = 16, fill_value = (mean of your dataset), mask_fill_value = None)
   - Achieve 85% accuracy with as many epochs as you want
   - Total number of parameters should be less than 200k
   - Upload the code to GitHub

2. Attempt S9-Assignment Solution.
3. Answer the following questions in the Assignment QnA:
   1. Copy and paste your model code from your model.py file (full code) [125]
   2. Copy and paste the output of torchsummary [125]
   3. Copy and paste the code where you implemented albumentation transformations for all three augmentations [125]
   4. Copy and paste your training log (including validation/testing after each epoch) [125]
   5. Share the link to your README.md file [200]

---

# **Solution**

Notebook file - `session_09.ipynb`

<br>

# **Key Points**

- The model consists of 4 blocks followed by a GAP layer.
- Instead of MaxPooling or a 3x3 Strided convolution, Dilated + Strided convolution is used in Blocks 2, 3, and 4.
- Depthwise Separable Convolution is used in Blocks 2 and 4.
- Dilated Convolution (Stride = 1) is used in Block 3.
- Output RF = 101

<hr>
<br>

# **RF Calculations**

| **Input Size** | **Input RF** | **Kernel (+Dilation)** | **Padding** | **Stride** | **Jump In** | **Jump Out** | **Output RF** | **Output Size** | **Block** | **Notes**                          |
| -------------- | ------------ | ---------------------- | ----------- | ---------- | ----------- | ------------ | ------------- | --------------- | --------- | ---------------------------------- |
| 32             | 1            | 5                      | 2           | 1          | 1           | 1            | 5             | 32              | 1         | Normal Conv                        |
| 32             | 5            | 3                      | 1           | 1          | 1           | 1            | 7             | 32              | 1         | Normal Conv                        |
| 32             | 7            | 3                      | 1           | 1          | 1           | 1            | 9             | 32              | 1         | Normal Conv                        |
| 32             | 9            | 5                      | 2           | 2          | 1           | 2            | 13            | 16              | 2         | Dilated + Strided Conv             |
| 16             | 13           | 3                      | 1           | 1          | 2           | 2            | 17            | 16              | 2         | Depthwise Separable Conv           |
| 16             | 17           | 3                      | 1           | 1          | 2           | 2            | 21            | 16              | 2         | Depthwise Separable Conv           |
| 16             | 21           | 5                      | 2           | 2          | 2           | 4            | 29            | 8               | 3         | Dilated + Strided Conv             |
| 8              | 29           | 5                      | 1           | 1          | 4           | 4            | 45            | 8               | 3         | Dilated Conv                       |
| 8              | 45           | 3                      | 1           | 1          | 4           | 4            | 53            | 8               | 3         | Normal Conv                        |
| 8              | 53           | 5                      | 2           | 2          | 4           | 8            | 69            | 4               | 4         | Dilated + Strided Conv             |
| 4              | 69           | 3                      | 1           | 1          | 8           | 8            | 85            | 4               | 4         | Depthwise Separable Conv           |
| 4              | 85           | 3                      | 1           | 1          | 8           | 8            | 101           | 4               | 4         | Normal Conv (Last conv before GAP) |

<hr>
<br>

# **Model Parameters**

```
============================================================================================================================================
Layer (type (var_name))                            Kernel Shape       Input Shape        Output Shape       Param #            Trainable
============================================================================================================================================
Net (Net)                                          --                 [2, 3, 32, 32]     [2, 10]            --                 True
├─ConvBlock (conv_block1)                          --                 [2, 3, 32, 32]     [2, 25, 32, 32]    --                 True
│    └─Sequential (conv_block)                     --                 [2, 3, 32, 32]     [2, 25, 32, 32]    --                 True
│    │    └─ConvLayer (0)                          --                 [2, 3, 32, 32]     [2, 25, 32, 32]    --                 True
│    │    │    └─Conv2d (conv_layer)               [3, 3]             [2, 3, 32, 32]     [2, 25, 32, 32]    675                True
│    │    │    └─BatchNorm2d (norm_layer)          --                 [2, 25, 32, 32]    [2, 25, 32, 32]    50                 True
│    │    │    └─ReLU (activation_layer)           --                 [2, 25, 32, 32]    [2, 25, 32, 32]    --                 --
│    │    └─ConvLayer (1)                          --                 [2, 25, 32, 32]    [2, 25, 32, 32]    --                 True
│    │    │    └─DepthwiseConv (conv_layer)        --                 [2, 25, 32, 32]    [2, 25, 32, 32]    900                True
│    │    │    └─BatchNorm2d (norm_layer)          --                 [2, 25, 32, 32]    [2, 25, 32, 32]    50                 True
│    │    │    └─ReLU (activation_layer)           --                 [2, 25, 32, 32]    [2, 25, 32, 32]    --                 --
├─TransBlock (trans_block1)                        --                 [2, 25, 32, 32]    [2, 32, 30, 30]    --                 True
│    └─ConvLayer (trans_block)                     --                 [2, 25, 32, 32]    [2, 32, 30, 30]    --                 True
│    │    └─Conv2d (conv_layer)                    [3, 3]             [2, 25, 32, 32]    [2, 32, 30, 30]    7,200              True
│    │    └─BatchNorm2d (norm_layer)               --                 [2, 32, 30, 30]    [2, 32, 30, 30]    64                 True
│    │    └─ReLU (activation_layer)                --                 [2, 32, 30, 30]    [2, 32, 30, 30]    --                 --
├─ConvBlock (conv_block2)                          --                 [2, 32, 30, 30]    [2, 32, 30, 30]    --                 True
│    └─Sequential (conv_block)                     --                 [2, 32, 30, 30]    [2, 32, 30, 30]    --                 True
│    │    └─ConvLayer (0)                          --                 [2, 32, 30, 30]    [2, 32, 30, 30]    --                 True
│    │    │    └─DepthwiseConv (conv_layer)        --                 [2, 32, 30, 30]    [2, 32, 30, 30]    1,376              True
│    │    │    └─BatchNorm2d (norm_layer)          --                 [2, 32, 30, 30]    [2, 32, 30, 30]    64                 True
│    │    │    └─ReLU (activation_layer)           --                 [2, 32, 30, 30]    [2, 32, 30, 30]    --                 --
│    │    └─ConvLayer (1)                          --                 [2, 32, 30, 30]    [2, 32, 30, 30]    --                 True
│    │    │    └─DepthwiseConv (conv_layer)        --                 [2, 32, 30, 30]    [2, 32, 30, 30]    1,376              True
│    │    │    └─BatchNorm2d (norm_layer)          --                 [2, 32, 30, 30]    [2, 32, 30, 30]    64                 True
│    │    │    └─ReLU (activation_layer)           --                 [2, 32, 30, 30]    [2, 32, 30, 30]    --                 --
├─TransBlock (trans_block2)                        --                 [2, 32, 30, 30]    [2, 65, 26, 26]    --                 True
│    └─ConvLayer (trans_block)                     --                 [2, 32, 30, 30]    [2, 65, 26, 26]    --                 True
│    │    └─Conv2d (conv_layer)                    [3, 3]             [2, 32, 30, 30]    [2, 65, 26, 26]    18,720             True
│    │    └─BatchNorm2d (norm_layer)               --                 [2, 65, 26, 26]    [2, 65, 26, 26]    130                True
│    │    └─ReLU (activation_layer)                --                 [2, 65, 26, 26]    [2, 65, 26, 26]    --                 --
├─ConvBlock (conv_block3)                          --                 [2, 65, 26, 26]    [2, 65, 26, 26]    --                 True
│    └─Sequential (conv_block)                     --                 [2, 65, 26, 26]    [2, 65, 26, 26]    --                 True
│    │    └─ConvLayer (0)                          --                 [2, 65, 26, 26]    [2, 65, 26, 26]    --                 True
│    │    │    └─DepthwiseConv (conv_layer)        --                 [2, 65, 26, 26]    [2, 65, 26, 26]    4,940              True
│    │    │    └─BatchNorm2d (norm_layer)          --                 [2, 65, 26, 26]    [2, 65, 26, 26]    130                True
│    │    │    └─ReLU (activation_layer)           --                 [2, 65, 26, 26]    [2, 65, 26, 26]    --                 --
│    │    └─ConvLayer (1)                          --                 [2, 65, 26, 26]    [2, 65, 26, 26]    --                 True
│    │    │    └─DepthwiseConv (conv_layer)        --                 [2, 65, 26, 26]    [2, 65, 26, 26]    4,940              True
│    │    │    └─BatchNorm2d (norm_layer)          --                 [2, 65, 26, 26]    [2, 65, 26, 26]    130                True
│    │    │    └─ReLU (activation_layer)           --                 [2, 65, 26, 26]    [2, 65, 26, 26]    --                 --
├─TransBlock (trans_block3)                        --                 [2, 65, 26, 26]    [2, 95, 18, 18]    --                 True
│    └─ConvLayer (trans_block)                     --                 [2, 65, 26, 26]    [2, 95, 18, 18]    --                 True
│    │    └─Conv2d (conv_layer)                    [3, 3]             [2, 65, 26, 26]    [2, 95, 18, 18]    55,575             True
│    │    └─BatchNorm2d (norm_layer)               --                 [2, 95, 18, 18]    [2, 95, 18, 18]    190                True
│    │    └─ReLU (activation_layer)                --                 [2, 95, 18, 18]    [2, 95, 18, 18]    --                 --
├─ConvBlock (conv_block4)                          --                 [2, 95, 18, 18]    [2, 95, 18, 18]    --                 True
│    └─Sequential (conv_block)                     --                 [2, 95, 18, 18]    [2, 95, 18, 18]    --                 True
│    │    └─ConvLayer (0)                          --                 [2, 95, 18, 18]    [2, 95, 18, 18]    --                 True
│    │    │    └─DepthwiseConv (conv_layer)        --                 [2, 95, 18, 18]    [2, 95, 18, 18]    10,070             True
│    │    │    └─BatchNorm2d (norm_layer)          --                 [2, 95, 18, 18]    [2, 95, 18, 18]    190                True
│    │    │    └─ReLU (activation_layer)           --                 [2, 95, 18, 18]    [2, 95, 18, 18]    --                 --
│    │    └─ConvLayer (1)                          --                 [2, 95, 18, 18]    [2, 95, 18, 18]    --                 True
│    │    │    └─DepthwiseConv (conv_layer)        --                 [2, 95, 18, 18]    [2, 95, 18, 18]    10,070             True
│    │    │    └─BatchNorm2d (norm_layer)          --                 [2, 95, 18, 18]    [2, 95, 18, 18]    190                True
│    │    │    └─ReLU (activation_layer)           --                 [2, 95, 18, 18]    [2, 95, 18, 18]    --                 --
├─TransBlock (trans_block4)                        --                 [2, 95, 18, 18]    [2, 95, 2, 2]      --                 True
│    └─ConvLayer (trans_block)                     --                 [2, 95, 18, 18]    [2, 95, 2, 2]      --                 True
│    │    └─Conv2d (conv_layer)                    [3, 3]             [2, 95, 18, 18]    [2, 95, 2, 2]      81,225             True
│    │    └─BatchNorm2d (norm_layer)               --                 [2, 95, 2, 2]      [2, 95, 2, 2]      190                True
│    │    └─ReLU (activation_layer)                --                 [2, 95, 2, 2]      [2, 95, 2, 2]      --                 --
├─Sequential (out_block)                           --                 [2, 95, 2, 2]      [2, 10]            --                 True
│    └─AdaptiveAvgPool2d (0)                       --                 [2, 95, 2, 2]      [2, 95, 1, 1]      --                 --
│    └─Conv2d (1)                                  [1, 1]             [2, 95, 1, 1]      [2, 10, 1, 1]      960                True
│    └─Flatten (2)                                 --                 [2, 10, 1, 1]      [2, 10]            --                 --
│    └─LogSoftmax (3)                              --                 [2, 10]            [2, 10]            --                 --
============================================================================================================================================
Total params: 199,469
Trainable params: 199,469
Non-trainable params: 0
Total mult-adds (M): 108.60
============================================================================================================================================
Input size (MB): 0.02
Forward/backward pass size (MB): 19.03
Params size (MB): 0.80
Estimated Total Size (MB): 19.86
============================================================================================================================================
```

<hr>
<br>

# **Results**

- Total Parameters = `199,469`
- Minimum training loss = `0.326763570%`
- Minimum testing loss = `0.003746615%`
- Best training accuracy = `80.09%`
- Best testing accuracy = `84.31%`


<hr>
<br>

# **Misclassified Images**

![Misclassified Images](./static/misclass.png "Misclassified Images")

<hr>
<br>

# **Loss and Accuracy Graphs**

![Loss and Accuracy Graphs](./static/graphs.png "Loss and Accuracy Graphs")
