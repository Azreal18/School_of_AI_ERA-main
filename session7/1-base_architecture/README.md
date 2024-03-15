### Analysis of Iteration 1
<I>

---


**Objective:**

*   Set up the basic code with data loaders, sample architecture, and predictions + validation.
*   No parameter checks or augmentations are implemented at this stage.


**Results:**<br>

*   Epochs - 15
*   Parameters - 6.3M
*   Best Train Accuracy -  99.86% (last epoch)
*   Best Test Accuracy - 99.14% (last epoch)
*   Difference between Train/Test Accuracy of the last layer - 0.72


**Analysis:**<br>


*   The training logs indicate signs of overfitting.
*   The model has a large capacity (parameter count), making it more complex.
*   Since no transformations are applied, the trained model becomes biased towards the training images, which may not accurately represent real-world conditions.


---
