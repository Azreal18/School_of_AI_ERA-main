# TSAI-ERAv2-S19
 
## Objective

1. Train a mini GPT model following the instruction from Andrej Karpathy in this [video](https://www.youtube.com/watch?v=kCc8FmEb1nY&t=2s)
2. Upload the model to HuggingFace Apps

## Dataset - tiny shakespeare, character-level
Tiny shakespeare, of the good old char-rnn fame :) Treated on character-level.

- Tokenization performed on Character level
- Vocab size 65. Following are the unique tokens
    - `!$&',-.3:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz`
- Number of total tokens - 1115394
    - trained on 1,003,854 tokens (90%)
    - validation is performed on 111,540 tokens (10%)



## Steps

### Initial experiment

1. Followed the [video](https://www.youtube.com/watch?v=kCc8FmEb1nY&t=2s) and created the [notebook](https://github.com/Azreal18/TSAI-ERAv2-S19/blob/main/dev.ipynb) for experiment.
2. Updated the model code for adding Gradio in the notebook as part of experiment
3. Added the Gradio app in the notebook



## The Huggingface Spaces Gradio App

The app is available [here](https://huggingface.co/spaces/Azreal18/gpt_test)