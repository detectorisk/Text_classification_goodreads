# Text_classification_goodreads
 goodreads_rating_prediction

#Overview:

The focus of this project is to predict the review scores on goodreads from just review text written by users. This kind of text classification (into 5 classes of scores) has been done through deep neural networks. These results, both show the significance of using a neural network for this problem, as well as the need to move toward more computationally complex and deep networks. 

We started with some descriptory analysis, to see general trends within our training data. Then, we built a basic machine learning text classification model using logistical regression. Having that as our ML baseline, we built a simple 3 layer neural network, using 1 dense layer, 1 embedding layer, and one output layer. We looked for accuracy levels in both test and train datasets, and checked if our results are robust, avoiding overfitting (as these models can often get). After this, we began investigating optimal number of neurons within this basic 3 layer neural network by checking model performance for varying neurons. 

From there, we built a more complex neural network, namely with a layer of a 1-D Convolutional network. Here again, we try to look for the "ideal" activation function in the output layer, though our research suggested that for multi-label classifications, softmax activation functions are often the [norm](https://machinelearningmastery.com/softmax-activation-function-with-python/). 

Lastly, we tried some deep neural networks such as LSTM to better our performance. Our initial observations were that these models were time and memory consuming, and allowed for less manipulation given our limited capacity of system availability. Further, we found these models to add very little to the overall accuracy over our test dataset. This perhaps points toward more preprocessing in our input data.

#Method:

1. Data exploration : Our initial finding was that the ratings are skewed in favour of high ratings (more than 3). However, the number of words used per review seem to be similar across all ratings. This helps us introduce Bag-of-words tokenisation methods to vectorize our text data into numeric form. 

2. Text pre-processing : As such there doesn't seem to be large outliers in the rating data column which need to be dealt with. As for the text data, as mentioned before, we used tokenization methods. For the ML baseline model we used the "CountVectorizer" to transform the data into numeric vectors. However, for all subsequent neural networks we used "Tokenizer" function from keras and then converted our tokens into sequences with a built in "text_to_sequence" function. 

  In the former method, the tokenization method converts all sentences into token vectors, and then takes them as features into the model. However, since we are dealing with review text, most often they do not have a lower or upper bound in terms of words, the number of features in this case for each input would differ. This can also bring potential bias into the model. To counter this, we have used the "Tokenizer", "text_to_sequence" and "pad_sequence" functions from keras. 

3. Train/Test Validation Strategy: Following general practice, for the ML baseline model we split all our labelled data (training dataset) into 80% training and 20% testing dataset. At this point, since we are using a baseline ML model, with no recourse for optimizing our accuracy levels, we didn't consider a validation dataset. 

  For all the neural networks models subsequently, we first split our labelled data in 80% training and 20% testing data. Of the training dataset thus obtained, we split it further into 75% actual training data, and 25% validation data. Here, the validation data is important to consider, as we will use it to validate our accuracy and loss results through each epoch, and our model wil try to optimize. 
