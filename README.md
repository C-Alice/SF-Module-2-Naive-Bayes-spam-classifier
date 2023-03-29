## Main task

The purpose of this project was to create and train a model, that would predict if the given text was spam or not.

## Theory

Spam filtering using Naive Bayes Classifier is one of the oldest and simplest ways to deal with spam.
Let's dive a little bit deeper into the method.

The Naive Bayes Classifier is based on Bayes' theorem.

<img src="https://miro.medium.com/max/768/1*zPseemLGYHMS8M0phAhhoA.png" width="300" height="200">

Where,

- P (A|B) - the probability that the email B belongs to the class A (spam);
- P (B|A) - probability of meeting an email B among all spam emails;
- P(A) - unconditional probability of meeting a spam email among all emails;
- P(B) - unconditional probability of the email B among emails.

To understand which class an email belongs to, we need to calculate probabilities for all classes and chose the class with the highest probability.

When we use Naive Bayes method, we presume that all the words in the text are independent (unlike in natural language).
That assumption allows us to approximate the conditional probability of the text (P(B|A)) by the product of the conditional probabilities of all the words included in the text.

Conditional probability of the word can be estimated as the number of times how much the word occurs in the texts belonging to a certain class divided to a number of all unique words.

Here the unknown words can be a problem, because they turn the overall text probability to 0. The typical way to avoid that is to add one to the frequency of each word, as if we met each word one time more.

To train the classifier we need a dataset, where spam and ham emails are labeled accordingly. After obtaining all the needed statistics in the learning process, the model is able to tell the class (spam or ham) of any given new text. Usually also a certain threshold is set: all the emails with spam probability above it get marked as spam.   


## Now you try

We created an app, that is based on the results of training our model on [Spam or Not Spam Dataset](https://www.kaggle.com/ozlerhakan/spam-or-not-spam-dataset). 

![](demo.gif)

You are welcome to check its performance.

To run the project, make sure you have Numpy, Pandas and Flask installed.

Clone the repository and execute the following commands in your terminal:

```
$ export FLASK_APP=run.py
$ python -m flask run
```
Then use the output localhost URL to open the app homepage.
Insert an email into text field and press "check". After you get a prediction, and it is not spam, you can read the email and report, if you think it actually was.
