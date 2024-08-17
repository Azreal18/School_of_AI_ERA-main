# TASI_ERAv2_S17

## Objective

1. Write the model.py file and then train the model such that it reaches a loss of 3.5 or less in as many epochs as you want. 


## Dataset - opus_books

- The dataset is sourced from [Helsinki-NLP/opus_books](https://huggingface.co/datasets/Helsinki-NLP/opus_books).
- This is a collection of copyright-free books aligned by Andras Farkas, which are available from [here](http://www.farkastranslations.com/bilingual_books.php).
- The dataset supports 16 different languages.


## Result

The loss went down to 2.790 at the 8th epoch but the code went on till 9 epochs before interruption where the loss was 3.140


## Training Logs

```
Max length of the source sentence: 309
Max length of the source target: 274

Processing Epoch 00: 100%|██████████| 1819/1819 [17:50<00:00,  1.70it/s, loss=5.358]
Processing Epoch 01: 100%|██████████| 1819/1819 [17:51<00:00,  1.70it/s, loss=5.269]
Processing Epoch 02: 100%|██████████| 1819/1819 [17:52<00:00,  1.70it/s, loss=5.004]
Processing Epoch 03: 100%|██████████| 1819/1819 [17:52<00:00,  1.70it/s, loss=4.539]
Processing Epoch 04: 100%|██████████| 1819/1819 [17:52<00:00,  1.70it/s, loss=3.825]
Processing Epoch 05: 100%|██████████| 1819/1819 [17:52<00:00,  1.70it/s, loss=4.104]
Processing Epoch 06: 100%|██████████| 1819/1819 [17:52<00:00,  1.70it/s, loss=3.818]
Processing Epoch 07: 100%|██████████| 1819/1819 [17:50<00:00,  1.70it/s, loss=2.790]
Processing Epoch 08: 100%|██████████| 1819/1819 [17:49<00:00,  1.70it/s, loss=3.140]
Processing Epoch 09:  32%|███▏      | 579/1819 [05:40<12:09,  1.70it/s, loss=2.878]
```

