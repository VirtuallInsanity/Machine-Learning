# Pneumonia recognition based on chest xray
Competition on kaggle https://www.kaggle.com/paultimothymooney/chest-xray-pneumoni \
For visualization decision making by neural network https://github.com/totti0223/gradcamplusplus

Model | Accuracy
------------ | -------------
VGG16 Fine Tune | 94,871%
VGG16 No weights | 92,948%
VGG16 Frozen | 87,019%
Model #1 (adam) | 88,621%
Model #1 (RMSprop) | 90,064%
Model #2 (adam) | 84,134%
Model #2 (RMSprop) | 80,929%
Model #3 (adam) | 85,416%
Model #3 (RMSprop) | 81,25%

# Visualization examples
Image of healty patient
![Normal](https://github.com/VirtuallInsanity/Machine-Learning/blob/pneumonia_xray/20eph_vgg16_frozen_adam0001/test_normal.png)
Image of healty patient
![Normal_image 2](https://github.com/VirtuallInsanity/Machine-Learning/blob/pneumonia_xray/20eph_vgg16_frozen_adam0001/test_normal_2.png)
Image of patient infected with viral pneumonia
![Viral_pneumonia](https://github.com/VirtuallInsanity/Machine-Learning/blob/pneumonia_xray/20eph_vgg16_frozen_adam0001/test_pneumonia_virus.png)
Image of patient infected with bacterial pneumonia
![Bacterial_Pneumonia](https://github.com/VirtuallInsanity/Machine-Learning/blob/pneumonia_xray/20eph_vgg16_frozen_adam0001/test_pneumonia_bacteria.png)
