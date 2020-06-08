## Main task

The purpose of this project was to write and train a model, that would predict if the given text was spam or not.

## Theory

Spam filtering using Bayes theorem is one of the oldest and simpliest ways to deal with spam.

Each word in a text has some probability to occur in either spam or ham email. When we use Naive Bayes method, we presume that all the words in the text are independent, and we can multiply their probabilities to get the "spaminess" of the whole email. Therefore, many words with high spam probability increase the probability of the text itself to be spam and vice versa.

For classifier to learn these probabilities we need a dataset, where spam and ham emails are labeled accordingly. After obtaining a particular probabilities set in the learning process, the model is able to tell the category (spam or ham) of any given new text. Usually a certain threshold is set for that: all the emails with probability above it get marked as spam.   


## Now you try

We created an app, that is based on the results of training our model on [Spam or Not Spam Dataset](https://www.kaggle.com/ozlerhakan/spam-or-not-spam-dataset). You are welcome to check its perfomance [here]()
