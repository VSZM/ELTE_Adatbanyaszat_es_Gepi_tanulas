
# Classification

Today we learn about classification. Let's load the [Iris dataset](https://scikit-learn.org/stable/datasets/index.html#iris-dataset) first. This time we will be using the sklearn [datasets package](https://scikit-learn.org/stable/datasets/index.html)>


```python
from sklearn import datasets

iris = datasets.load_iris()
```

Let's split the data into train and test sets>


```python
from sklearn.model_selection import train_test_split

X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
```

First we try the [K Nearest Neighbor classifier](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html). The single most important hyper parameter of this model is the K value of neighbors to consider when classifying a new object. Let's set it to 3 for now.>


```python
from sklearn.neighbors import KNeighborsClassifier

model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)
```




    KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
                         metric_params=None, n_jobs=None, n_neighbors=10, p=2,
                         weights='uniform')



After fitting the model we can print a report about the performance of our model using [Classificaton Report](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html)>


```python
from sklearn.metrics import classification_report

y_pred = model.predict(X_test)

print(classification_report(y_test, y_pred))
```

                  precision    recall  f1-score   support
    
               0       1.00      1.00      1.00        10
               1       1.00      1.00      1.00         9
               2       1.00      1.00      1.00        11
    
        accuracy                           1.00        30
       macro avg       1.00      1.00      1.00        30
    weighted avg       1.00      1.00      1.00        30
    
    

As we can see this problem was very easy, so we could achive 100 % precision on it. 

Let's load a bit more complicated dataset, the [Breast Cancer dataset](https://scikit-learn.org/stable/datasets/index.html#breast-cancer-dataset)>


```python
breast_cancer = datasets.load_breast_cancer()
print(breast_cancer.DESCR)
```

    .. _breast_cancer_dataset:
    
    Breast cancer wisconsin (diagnostic) dataset
    --------------------------------------------
    
    **Data Set Characteristics:**
    
        :Number of Instances: 569
    
        :Number of Attributes: 30 numeric, predictive attributes and the class
    
        :Attribute Information:
            - radius (mean of distances from center to points on the perimeter)
            - texture (standard deviation of gray-scale values)
            - perimeter
            - area
            - smoothness (local variation in radius lengths)
            - compactness (perimeter^2 / area - 1.0)
            - concavity (severity of concave portions of the contour)
            - concave points (number of concave portions of the contour)
            - symmetry 
            - fractal dimension ("coastline approximation" - 1)
    
            The mean, standard error, and "worst" or largest (mean of the three
            largest values) of these features were computed for each image,
            resulting in 30 features.  For instance, field 3 is Mean Radius, field
            13 is Radius SE, field 23 is Worst Radius.
    
            - class:
                    - WDBC-Malignant
                    - WDBC-Benign
    
        :Summary Statistics:
    
        ===================================== ====== ======
                                               Min    Max
        ===================================== ====== ======
        radius (mean):                        6.981  28.11
        texture (mean):                       9.71   39.28
        perimeter (mean):                     43.79  188.5
        area (mean):                          143.5  2501.0
        smoothness (mean):                    0.053  0.163
        compactness (mean):                   0.019  0.345
        concavity (mean):                     0.0    0.427
        concave points (mean):                0.0    0.201
        symmetry (mean):                      0.106  0.304
        fractal dimension (mean):             0.05   0.097
        radius (standard error):              0.112  2.873
        texture (standard error):             0.36   4.885
        perimeter (standard error):           0.757  21.98
        area (standard error):                6.802  542.2
        smoothness (standard error):          0.002  0.031
        compactness (standard error):         0.002  0.135
        concavity (standard error):           0.0    0.396
        concave points (standard error):      0.0    0.053
        symmetry (standard error):            0.008  0.079
        fractal dimension (standard error):   0.001  0.03
        radius (worst):                       7.93   36.04
        texture (worst):                      12.02  49.54
        perimeter (worst):                    50.41  251.2
        area (worst):                         185.2  4254.0
        smoothness (worst):                   0.071  0.223
        compactness (worst):                  0.027  1.058
        concavity (worst):                    0.0    1.252
        concave points (worst):               0.0    0.291
        symmetry (worst):                     0.156  0.664
        fractal dimension (worst):            0.055  0.208
        ===================================== ====== ======
    
        :Missing Attribute Values: None
    
        :Class Distribution: 212 - Malignant, 357 - Benign
    
        :Creator:  Dr. William H. Wolberg, W. Nick Street, Olvi L. Mangasarian
    
        :Donor: Nick Street
    
        :Date: November, 1995
    
    This is a copy of UCI ML Breast Cancer Wisconsin (Diagnostic) datasets.
    https://goo.gl/U2Uwz2
    
    Features are computed from a digitized image of a fine needle
    aspirate (FNA) of a breast mass.  They describe
    characteristics of the cell nuclei present in the image.
    
    Separating plane described above was obtained using
    Multisurface Method-Tree (MSM-T) [K. P. Bennett, "Decision Tree
    Construction Via Linear Programming." Proceedings of the 4th
    Midwest Artificial Intelligence and Cognitive Science Society,
    pp. 97-101, 1992], a classification method which uses linear
    programming to construct a decision tree.  Relevant features
    were selected using an exhaustive search in the space of 1-4
    features and 1-3 separating planes.
    
    The actual linear program used to obtain the separating plane
    in the 3-dimensional space is that described in:
    [K. P. Bennett and O. L. Mangasarian: "Robust Linear
    Programming Discrimination of Two Linearly Inseparable Sets",
    Optimization Methods and Software 1, 1992, 23-34].
    
    This database is also available through the UW CS ftp server:
    
    ftp ftp.cs.wisc.edu
    cd math-prog/cpo-dataset/machine-learn/WDBC/
    
    .. topic:: References
    
       - W.N. Street, W.H. Wolberg and O.L. Mangasarian. Nuclear feature extraction 
         for breast tumor diagnosis. IS&T/SPIE 1993 International Symposium on 
         Electronic Imaging: Science and Technology, volume 1905, pages 861-870,
         San Jose, CA, 1993.
       - O.L. Mangasarian, W.N. Street and W.H. Wolberg. Breast cancer diagnosis and 
         prognosis via linear programming. Operations Research, 43(4), pages 570-577, 
         July-August 1995.
       - W.H. Wolberg, W.N. Street, and O.L. Mangasarian. Machine learning techniques
         to diagnose breast cancer from fine-needle aspirates. Cancer Letters 77 (1994) 
         163-171.
    


```python
X = breast_cancer.data
y = breast_cancer.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
```

After splitting the data we create a [Logistic Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) model this time> 


```python
from sklearn.linear_model import LogisticRegression

model2 = LogisticRegression(max_iter = 10000)
model2.fit(X_train, y_train)
```




    LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                       intercept_scaling=1, l1_ratio=None, max_iter=10000,
                       multi_class='auto', n_jobs=None, penalty='l2',
                       random_state=None, solver='lbfgs', tol=0.0001, verbose=0,
                       warm_start=False)




```python
y_pred = model2.predict(X_test)

print(classification_report(y_test, y_pred))
```

                  precision    recall  f1-score   support
    
               0       0.97      0.91      0.94        43
               1       0.95      0.99      0.97        71
    
        accuracy                           0.96       114
       macro avg       0.96      0.95      0.95       114
    weighted avg       0.96      0.96      0.96       114
    
    

As we can see this problem is also quite easy. We didn't even have to include polynomial synthetic features, a simple linear regression solved the problem very well.

Nevertheless let us try KNN on this problem as well and see how it performs with K = 5.


```python
model3 = KNeighborsClassifier(n_neighbors=5)
model3.fit(X_train, y_train)

```




    KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
                         metric_params=None, n_jobs=None, n_neighbors=5, p=2,
                         weights='uniform')




```python
y_pred = model3.predict(X_test)

print(classification_report(y_test, y_pred))
```

                  precision    recall  f1-score   support
    
               0       1.00      0.88      0.94        43
               1       0.93      1.00      0.97        71
    
        accuracy                           0.96       114
       macro avg       0.97      0.94      0.95       114
    weighted avg       0.96      0.96      0.96       114
    
    

However the results are already very good we can try and optimize the hyperparameters of our model.

Scikit Learn provides tools for this, one of the most commonly used is [Grid Search](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html). This is an exhaustive search implementation that will try out all the given possible parameters and see which performs the best.>


```python
from sklearn.model_selection import GridSearchCV

hyper_params = {'n_neighbors': range(1,16), 'p':range(1,5)}
grid = GridSearchCV(KNeighborsClassifier(), hyper_params)
grid.fit(X_train, y_train)
```




    GridSearchCV(cv=None, error_score=nan,
                 estimator=KNeighborsClassifier(algorithm='auto', leaf_size=30,
                                                metric='minkowski',
                                                metric_params=None, n_jobs=None,
                                                n_neighbors=5, p=2,
                                                weights='uniform'),
                 iid='deprecated', n_jobs=None,
                 param_grid={'n_neighbors': range(1, 16), 'p': range(1, 5)},
                 pre_dispatch='2*n_jobs', refit=True, return_train_score=False,
                 scoring=None, verbose=0)



The best parameter settings are stored in the grid object's `best_params_` attribute>


```python
grid.best_params_
```




    {'n_neighbors': 6, 'p': 1}




```python
model4 = KNeighborsClassifier(n_neighbors=grid.best_params_['n_neighbors'], p = grid.best_params_['p'])
model4.fit(X_train, y_train)

y_pred = model4.predict(X_test)

print(classification_report(y_test, y_pred))
```

                  precision    recall  f1-score   support
    
               0       0.97      0.91      0.94        43
               1       0.95      0.99      0.97        71
    
        accuracy                           0.96       114
       macro avg       0.96      0.95      0.95       114
    weighted avg       0.96      0.96      0.96       114
    
    

# **Exercise**

1. Load the fuel Dataset and try to predict the used fuel type based on arbitrary predictor attributes and your choice of model! Try to optimize the hyperparameters of your model if needed! (Don't forget to label encode the target attribute and do numerical transformation of predictors where applicable)
2. Like in the previous task try to predict the air conditioner attribute of the duel dataset! 
3. Load the wine_dataset from sklearn datasets and try to predict the type of wine!
https://scikit-learn.org/stable/datasets/index.html#wine-dataset
