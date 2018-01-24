# LSTM-CNN-for-Sentiment-Analysis
## Motivation 
I read Convolutional Neural Networks for Sentence Classification https://arxiv.org/pdf/1408.5882.pdf, and think adding LSTM before CNN may achieve better performance, then I did this project
## Model
![screen shot 2018-01-23 at 11 36 08 pm](https://user-images.githubusercontent.com/20760190/35314759-3d753056-0096-11e8-9282-67601fe00b81.png)
## Results
![screen shot 2018-01-23 at 11 36 19 pm](https://user-images.githubusercontent.com/20760190/35314760-3d84354c-0096-11e8-900c-4c0e63e7e533.png)<br>
Note that here for simplicity, we did not apply Google word2vec, so the results should be worse than the original paper, but easier to implement :)
## File Description
1. lstm_cnn repo stores codes for our model
2. cnn repo stores pytorch version of implementation of Convolutional Neural Networks for Sentence Classification, for tensorflow version you can find it https://github.com/dennybritz/cnn-text-classification-tf 

## How to run
1. enter lstm_cnn repo
2. python lstm_cnn.py (without crossvalidation)
   python lstm_cnn_cv.py (for crossvalidation)
