# Yelp Restuarant Review Challenge

**Authors**: [Alan Wang](https://github.com/alanwmy00), [Hongyi Yang](https://github.com/hoy007/), Jiayu Hu

[**Report**](https://github.com/alanwmy00/YelpRestaurantReviewChallenge/blob/main/Report/Yelp%20Restaurant%20Review%20Challenge.pdf)

This report considers the rating prediction problem from review texts based on the Yelp challenge dataset. Beginning with a simple baseline model, we explore various classification approaches. For the random forest model, we extract features via sentiment lexicon; while for the gradient boosting classifier, such work is done by TF-IDF. In addition, we explore some higher-end deep learning models: we not only build a simple neural network classifier after preprocessing the review text with bag-of-bigrams and TF-IDF, but also apply a pretrained BERT classifier. Each of the above-mentioned models is fit on the training subset and evaluated on the testing subset with three metrics: accuracy, $F$ score, and $R^2$ score.
