## Session-7
<hr>

**Iteration-1**

(Params - 6.3M) Base architecture (no limit on parameters) - No augmentation, BN, regularization, Variable LR
Target:
* Basic code setup with Data Loaders / Sample Architecture / Predictions + Validation
* No check of parameters or augmenters taken care at this step

Results:
* Epochs - 15
* Parameters - 6.3M
* Best Train Acc - 99.86% (LAST epoch)
* Best Test Acc - 99.14% (LAST epoch)
* Train/Test Acc last layer - (Difference - 0.72)

Analysis:
* Training logs show signs of overfitting.
* The model is large in terms of capacity (parameter count), making it more complex.
* Since there are no transforms being done, the trained model becomes biased towards train images, which might not be an actual representation of true conditions.

**Iteration-2**

Target:
1. Restructuring the architecture with Conv block and transition blocks.
2. Make the model a bit lighter on trainable parameters as the data is MNIST which is very small.

Results:
1. Parameters: 54,416
2. Best Train Acc: 99.00%
3. Best Test Acc: 98.89%
4. Difference: 0.11
5. Epoch - 15

Analysis:
1. The model still overfits with a significant gap. Regularization needs to be introduced.
2. The model is less complex with a reduction in kernel values.
3. After reducing the intermediate kernel values, the model still performs well.

**Iteration-3**

Target:
1. Training the model with fewer iterations and restructuring the architecture for fewer trainable parameters or complexity.

Results:
1. Parameters: 10,546
2. Best Train Acc: 98.99% (15th epoch)
3. Best Test Acc: 98.84% (15th epoch)
4. Difference: Train accuracy is slightly higher, indicating a small overfit scenario
5. Epoch - 15

Analysis:
1. The model appears to be slightly overfit.
2. The architecture looks relatively simple overall and can be improved upon to achieve better accuracy and prevent overfitting.

**Iteration-4**

Target:
1. Introduce Batch Normalization to increase the model's efficiency and help it converge faster. Also, it provides some regularization.

Results:
1. Parameters: 6,288
2. Best Train Acc: 99.69% (last epoch)
3. Best Test Acc: 99.20% (last epoch)
4. Difference: Train accuracy is higher
5. Epoch - 15

Analysis:
1. The model again overfits compared to the previous model.
2. Dropout needs to be introduced to bridge the gap between train and test accuracy.

**Iteration-5**

Target:
1. Introduce Dropout to decrease overfitting.
2. Apply different dropout rates as per kernels in each conv layer.

Results:
1. Parameters: 6,288
2. Best Train Acc: 99.22%
3. Best Test Acc: 99.34%
4. Difference: Test accuracy is higher
5. Epoch - 15

Analysis:
1. The model seems to be stable with no signs of overfitting.
2. Model accuracy can be further improved by tuning learning rates or changing the architecture to include more feature extractors.

**Iteration-6**

Target:
1. Improve model accuracy to achieve a stable test accuracy above 99.4
2. Added Global Average Pooling (GAP) to remove the last layer

Results:
1. Parameters: 5,328
2. Best Train Acc: 99.19% (last Epoch)
3. Best Test Acc: 99.22% (last epoch)
4. Difference: No overfitting
5. Epoch - 15

Analysis:
1. Model test accuracy is consistently around 99.20, which needs to be further improved by adding augmentation.

**Iteration-7**

Target:
1. Improve model accuracy to achieve a stable test accuracy above 99.4
2. Use LR Scheduler to apply different learning rates to each epoch.
3. Add augmentation to enable the model to learn from different test data.
4. Change the architecture to make it more complex.

Results:
1. Parameters: 7,592
2. Best Train Acc: 98.90% (last epoch)
3. Best Test Acc: 99.47% (12th epoch) / 99.45% (13th epoch) / 99.41% (14th epoch) / 99.44% (15th epoch)
4. Difference: No overfitting
5. Epoch - 15

Analysis:
1. The model consistently achieves a test accuracy above 99.40, which is promising.
2. There is a slight underfitting of the model as the test score is high.
