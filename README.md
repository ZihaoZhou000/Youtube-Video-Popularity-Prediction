# Youtube-Video-Popularity-Prediction

## Introducion
Youtube is the world-famous video sharing website platform that allows people to upload all kinds of videos. 
There are so many video uploaded every day, but the time of users is limited. Therefore, a good title is crucial. 
Most users makes the decision on whether to click on the vedio at the first glimpse on the title.

In this project, a method to represent the video popularity is introduced, and a deep learning model to predict the popularity video scores based on the title, tags and disciption is developed. which will help content creater to choose the best title that attract more viewers and generate more revenue from sponsored companies or Youtube itself.
The prediction model are created to output the probability distributions instead of a single value of the target to handle heteroscedastic uncertainty.

## Data Preprocessing
The dataset used for this project is the trending YouTube video statistics[1]. 
After dropping non-English data and replicate data, the rest of the data is randomly split into 20000 training data and 3872 test data. 
The titles, tags and descriptions are tokenized, puctuation-removed and transformed into numeric sequences. The sequences are then padded to the same length. 
A method referenced from a Stanford article[2] is used to calculate the popularity score, which will be the target of the supervised learning, based on views, comments, likes and dislikes. 

## Model Construction
When it comes to predictive modeling, aleatoric uncertainty rises when, for data in the training set, data points with very similar feature vectors have targets with substantially different values. 
No matter how good the model is at fitting the average trend, it won’t be able to fit every datapoint perfectly when aleatoric uncertainty is present. 
If this uncertainty stays constant for different x-values, it is called homoscedastic uncertainty; if this uncertainty changes for different x-values in the feature space, it is called heteroscedastic uncertainty.[3]

A standard neural network can only handle homoscedastic uncertainty by introducing bias, which is independent with x-values. 
In order to deal with heteroscedastic uncertainty, our general approach is to train a model that splits the neural network into two streams, mean and standard deviation, and use them to parameterize a distribution.
As the graph shows, the neural network takes 3 sequence inputs, title, tags and description, and each of them passes through a masking layer to mask zero elements that are used to pad the sequence to the same length. 
Then, 3 streams of sequences with different length are respectively embedded into 50 dimensional vectors using pre-trained GloVe. 
After that, the sequences pass through a LSTM layer that outputs a 8 dimensional vector for each sequence and all 3 streams are concatenated into one. 
Finally, the concat layer is split into two streams for mean and standard deviation, each of which is a 2-layer feed-forward network. 
Linear function is used as the output function for mean since the score could be negative when likes are less than 1.5*dislikes; softplus function is used as the output function for standard deviation, since it should be greater than 0. 
Assuming that the output mean and standard deviation parameterize either a Gaussian or a Laplace (double exponential) distribution, the loss could then be defined as the negative log likelihood of the target under the distribution.

At first, the heteroscedastic model drops into a local optimum after a few epochs that outputs a large standard deviation and a mean far from the true value for each data. 
To address this issue, a homoscedastic model is bulit, the neural net of which stays exactly the same as the heteroscedastic model for the most parts except it doesn’t have the streams for standard deviation and therefore requires a different loss function. 
Here huber loss is used, which is more tolerant to the outliers and thus more suitable for the NLP task compared to regular RMSE. 

Then, we use the weights for the LSTM and feed-forward layers of the homoscedastic model as the initial points for the weights for the same layers of the heteroscedastic model to train for both Gaussian and Laplace distribution assumptions. 

## Results
Figure X1 and X2 shows the predicted Gaussian and Laplace distribution for the popularity score of a specific test data, where the x-axis is the popularity score,  y-axis is the negative log likelihood. 
The red line stands for the true popularity score and the predicted distribution of popularity score is in blue. 
The deep blue area is the 95% confidence interval of the predicted popularity score distribution. 
If the red line falls into the deep blue area, it is considered as an acceptable prediction. 
![Gaussian](https://user-images.githubusercontent.com/54859964/122877427-96e74180-d304-11eb-9d4f-851867a0520d.png)
![Laplace](https://user-images.githubusercontent.com/54859964/122877432-98b10500-d304-11eb-9595-ede5896bbf6e.png)
Overall, compared to the model with Gaussian distribution, the model with Laplace distribution assumption has a much smaller loss (negative log likelihood) on both training and test data, suggesting the latter is a better model, probably because the Laplace distribution has a heavy tail and therefore more tolerant to the outliers. 
Moreover, 87.59% of the test data has their true value in the 95% confidence interval of the predicted Laplace distribution, while only 34.91% true value of the test data fall into the 95% confidence interval of the predicted Gasssian distribution, which also shows that the model with Laplace distribution assumption is more suitable of this task. 
