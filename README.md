# Hybrid Recommendation System for Yelp Dataset
The dataset used in the project is from yelp. The link can be found here: https://www.yelp.com/dataset

It is a recommendation system built with yelp's dataset, which will predict user's ratings and responses for the business and shops in the dataset. Potential application of a similar model includes making recommendations of services and products the company provide that can best meet customer's interests and needs.

For this project, PCA is implemented to reduce dimensionality and top 10 principal components are kept as features. Also, in addition to the model-based recommendation system, which utilizes tuned xgboost regression model, a feature-agumented collaborative filtering is implemeneted and the result of it is added as a feature to future improve the performance of the model.

Error Distribution:
>=0 and <1: 105773
>=1 and <2: 34123
>=2 and <3: 6310
>=3 and <4: 790
>=4: 0

RMSE: 0.975

