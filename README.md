## Email Spam Filter in Go 
- Uses Naive Bayes algorithm with log-space calculations for accurate email spam detection
- Core Logic: This program implements a Naive Bayes spam filter that learns to distinguish spam from legitimate emails by analyzing word patterns.

### How it works:
#### Training Phase:
- Reads thousands of example emails labeled as "ham" (good) and "spam" (bad)
- Counts how often each word appears in both categories
- Builds a "vocabulary" of common words with their frequencies

#### Classification Phase:
- For any new email, it breaks it down into individual words
- Calculates probability that the email is spam vs ham based on word patterns
- Uses Bayesian probability: "If these words appear, how likely is it spam?"
- Chooses the higher probability as the prediction

#### Key Techniques:
- Bag of Words: Treats emails as unordered collections of words
- Log Probabilities: Prevents mathematical errors with very small numbers
- Threshold Filtering: Ignores rare words that don't help with classification
- Training Data: Uses the famous Enron email dataset for learning

- Simple Example: If an email contains words like "FREE", "WINNER", "MONEY" frequently seen in spam, it gets classified as spam. If it contains "meeting", "project", "schedule" from normal emails, it's marked as ham.

## Quick Start

```bash
git clone https://github.com/39sanskar/email-spam-filter.git
go build main.go
go run main.go
```

## References:
[Naive Bayes classifier](https://en.wikipedia.org/wiki/Naive_Bayes_classifier)

[Data-Set](https://www2.aueb.gr/users/ion/data/enron-spam/)

[log-space calculations](https://en.wikipedia.org/wiki/Log-space_reduction)

## Training data organized like this

```bash
./enron1/
  ham/
    ...
  spam/
    ...
...
./enron6/
  ham/
    ...
  spam/
    ...
```
