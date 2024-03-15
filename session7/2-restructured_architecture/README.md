
### Analysis of Iteration 2


---

**Target:**
1. Restructure the architecture with Conv blocks and transition blocks.
2. Reduce the number of trainable parameters for the MNIST dataset.

**Results:**
1. Parameters: 54,416
2. Best Train Accuracy: 99.00%
3. Best Test Accuracy: 98.89%
4. Difference: 0.11
5. Epochs: 15

**Analysis:**
1. The model still overfits with a significant gap. Regularization is needed.
2. The model is less complex with reduced kernel values.
3. Even after reducing the intermediate kernel values, the model still performs well.
