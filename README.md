# Data Mining Playground  

This repository contains five mini projects covering several main topics in Data Mining. Below you can find the list of projects:  
- [Iris Analysis](https://github.com/zahrasalarian/Data-Mining-Playground#iris-analysis)
- [Classification](https://github.com/zahrasalarian/Data-Mining-Playground#classification)
- [Association Rules](https://github.com/zahrasalarian/Data-Mining-Playground#association-rules)
- [Clustering](https://github.com/zahrasalarian/Data-Mining-Playground#clustering)
- [Diabetes Classifier](https://github.com/zahrasalarian/Data-Mining-Playground#diabetes-classifier)

## Iris Analysis

The aim of this project is to implement some preprocessing techniques to demonstrate the importance of understanding, cleaning, and adjusting the raw dataset. considered facets include:  
1. The importance of missing values
2. Non-numerical data
3. Normalization
4. PCA
5. Plotting

#### About the dataset  
The Iris flower dataset is a multivariate dataset consisting of 3 different types of irises’ (Setosa, Versicolour, and Virginica) petal and sepal length and width, stored in a 150x4 numpy.ndarray.  

[scikit-learn](https://scikit-learn.org/stable/) and [pandas](https://pandas.pydata.org) libraries were used to employ the techniques.  

## Classification  

In this project, the attempt is to classify data using neural networks with different architectures. The data is a large circle containing a smaller circle in 2d ([make_circles](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_circles.html)), from [scikit-learn](https://scikit-learn.org/stable/) library. The process of obtaining the best architecture for NN is as follows:  
1. A NN without activation funtions in its layeres
2. A NN with linear activation funtions
3. Employing a proper Mean Squared Error to the network
4. A single hidden layer NN with 16 nodes
5. Finding an adequet value for learning rate (lr = 0.01) by testing various values.
6. Design a sufficient NN to get the best result

In the next step, the [fashion_mnist](https://www.tensorflow.org/datasets/catalog/fashion_mnist) dataset was loaded from the [TensorFlow](https://www.tensorflow.org) library that's also been used to implement the NNs. This dataset was trained on the NN, and the results were exhibited through a Confusion Matrix.  

## Association Rules

This mini-project contains the implementations of Association Rules extraction. The process consists of three main parameters:
- Support: shows the popularity of an item according to the number of times it appears in transactions
- Confidence: shows the probability of buying item y if item x is bought. x -> y
- Lift

The last one is calculated from the combination of the first two ones through the below equation:  

![tf-idf equation](lift-equation.png?raw=true)  

This project employs the Apriori algorithm to implement association rules using [mlxtend](https://github.com/rasbt/mlxtend) library.

## Clustering

This project aims to cluster the [make_blobs](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_blobs.html) dataset using the K-means algorithm and elbow technique to find the best value for K. It also contains some complex clustering examples with illustrations. In the following, the DBSCAN algorithm is implemented, together with estimating the values for epsilon (for KNN) and MinPts.

## Diabetes Classifier

In this project, a Diabetes Classifier was developed to predict whether a new given case is diabetic. The used dataset consists of more than 70,000 records of patients who have filled out the questionnaire designed by the Centers for Disease Control and Prevention (CDC). It has 22 columns listed below:  
- Diabetes_binary: The target column that determines whether a person has diabetes or pre-diabetes  
- HighBP
- High Cholesterol
- Cholesterol Check
- BMI
- Smoker
- Stroke
- HeartDiseaseorAttack
- Physical Activity
- Fruits
- Veggies
- Heavy Alcohol Consumption
- Any Health Care
- No Doctor because of Cost
- General Health
- Mental Health
- Physical Health
- Difficulty Walking
- Sex
- Age
- Education
- Income

The [XGBoost](https://xgboost.readthedocs.io/en/stable/) or Extreme Gradient Boost was used to implement the classifier. But before moving on to that, the preprocessing steps are listed in the following:  
- Handle Missing Values 
    - Impute missing continuous values with Mean
    - Impute missing categorical values with the most frequent category
- Replace white spaces in columns' names with '_'
- Normalizing/Scaling
- One-hot-encoding

In the next step, we design and train an XGBoost classifier with such architecture as below:  

```
model = XGBClassifier(
            learning_rate=0.1, 
            max_depth=4, 
            n_estimators=200, 
            subsample=0.5, 
            colsample_bytree=1, 
            random=123, 
            eval_metric='auc', 
            verbosity=1, 
            tree_method='gpu_hist', 
            early_stop=10)
```
And finally, to obtain the best combination of hyperparameters, we employ GridSearchCV on the following values for each parameter to tune them:  
```
grid_params = {
    'learning_rate_list': [0.02, 0.05, 0.1, 0.3],
    'max_depth_list': [2, 3, 4],
    'n_estimators_list': [100 ,200 ,300],
    'colsample_bytree': [0.8 ,1]
}
```