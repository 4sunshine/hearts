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
unet_cuda_all_aug UNET GELU + RandCrop + RandNoise + Random_Refl: 0.7976 epoch 184 
unet_cuda_no_aug UNET GELU + No aug:0.7157 epoch 352
unet_cuda_all_aug UNET ReLU + No aug:0.68 epoch ...  
unet_cuda_relu_aug UNET ReLU + RandCrop + RandNoise + Random_Refl: 0.7984 epoch 390  
unet_cuda_relu_aug UNET GeLU + RandCrop + RandNoise + Random_Refl + SigmaCrop: 0.8079 epoch 260 
unet_cuda_gelu_sigma_crop_no_refl UNET GeLU + RandCrop + RandNoise + SigmaCrop: 0.7616 eoich 365 
crnn_gelu_sigma_crop_no_refl CRNN + RandCrop + RandomCrop + RandomNoise: 0.789 epoch 114  
unet_cuda_gelu_sigma_crop_slope UNET GeLU + RandCrop + RandNoise + SigmaCrop + Random_Refl + RandSlope0.1: 0.8039 eoich 253
unet_cuda_gelu_sigma_crop_slope UNET GeLU + RandCrop + RandNoise + SigmaCrop + Random_Refl + RandSlope0.3: 0.8095 eoich 296  
unet_cuda_gelu_sigma_crop_slope UNET GeLU + RandCrop + RandNoise + SigmaCrop + Random_Refl + RandSlope0.5: 0.8005 eoich 254  
unet_cuda_gelu_sigma_crop_slope UNET GeLU + RandCrop + RandNoise + SigmaCrop + Random_Refl + RandSlope0.3 + RandFlip: 0.7931 eoich 254  
unet_features UNET GeLU + RandCrop + RandNoise + SigmaCrop + Random_Refl + RandSlope0.3:  0.7968 eoich 398  
unet_features_noslope UNET GeLU + RandCrop + RandNoise + SigmaCrop + Random_Refl:  0.7941 eoich 345  
unet_features_nonorm_noslope UNET GeLU + RandCrop + RandNoise + SigmaCrop + Random_Refl + RandSlope0.3:  0.7932 eoich 345  

