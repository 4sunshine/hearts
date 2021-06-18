Default CRNN + Final ReLU + Random Crop: 0.8867 Epoch 96  
Default CRNN + Final Leak + Random Crop: 0.8618 Epoch 172  
Default CRNN + Final ReLU: 0.8341 Epoch 172  
Default CRNN (Baseline): 0.8832 Epoch 147  
Default CRNN CUDA (Baseline) + Random Crop: 0.8692 Epoch 101 <- HAS EXPLODING GRADIENTS  
Default CRNN (Baseline) AdamW + Random Crop: 0.8498 Epoch 63  
Default CRNN + Final ReLU + (Dropout) + Random Crop: 0.884 Epoch 86  
Default CRNN (Baseline) + (Dropout) + Random Crop: 0.8827 Epoch 179 -> NEED TO TRAIN MORE THAN 200  
1LSTM CRNN + Random Crop: 0.8522 Epoch 95  
Default CRNN + Final ReLU + RandomCrop + RandomNoise: 0.8824 epoch 97  
kek_gelu!!! Default 3GELU CRNN + Final ReLU + RandomCrop + RandomNoise: 0.8946 epoch 140  

Default 3GELU CRNN + Final Tanh + RandomCrop + RandomNoise: don't converge  

BINARY SCORES -> CORRELATES WITH AVERAGE
kek_gelu!!! Default 3GELU CRNN + Final ReLU + RandomCrop + RandomNoise: 0.8098 epoch 140
kek_gelu!!! Default 3GELU CRNN + Final ReLU + RandomCrop + RandomNoise + Random Reflect: 0.7837 epoch 140  

