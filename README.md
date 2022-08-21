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





## Clustering




## Diabetes Classifier