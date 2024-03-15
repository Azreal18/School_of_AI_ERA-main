## Analysis of Iteration 4

**Target:**
The goal of this code is to introduce Batch Normalization to improve the model's efficiency and convergence speed. Batch Normalization also provides some regularization.

**Results:**
Here are the results obtained from running the code:
1. Parameters: 6,288
2. Best Train Accuracy: 99.69% (last epoch)
3. Best Test Accuracy: 99.20% (last epoch)
4. Difference: The train accuracy is higher than the test accuracy.
5. Epochs: 15

**Analysis:**
Upon analysis, the following observations can be made:
1. The model is overfitting compared to the previous model.
2. To address this issue, it is recommended to introduce Dropout to bridge the gap between the train and test accuracy.
