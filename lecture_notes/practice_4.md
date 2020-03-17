# Data Preprocessing

In this practical we will be using a new dataset, the fuel dataset with errors. The dataset can be found in the [github repository](https://github.com/VSZM/ELTE_Adatbanyaszat_es_Gepi_tanulas/tree/master/practice)

Load the data>


```python
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
%matplotlib inline


df = pd.read_csv('practice/fuel_data_with_errors.txt', delimiter='\t')
# This option will get rid of the scientific notation
#pd.set_option('display.float_format', lambda x: '%.3f' % x)
df.head()
```


### **Data Quality**

Check if there are any missing values in *starttemp*>

```python
df['starttemp'].isnull().sum()
```

Add the mean value where *starttemp* is missing>

```python
df['starttemp'][df['starttemp'].isnull()] = df['starttemp'].mean()
```

A more advanced use case is to use other attributes of the row to estimate the missing attribute. For example estimate the missing distances traveled based on the traveled route>

```python
# creating a lookup table for distances based on the traveled route
route_to_distance = df.groupby('route')['dist'].mean()
   
distances = []
# iterating through all the rows and checking if any of the distances are missing. 
for row in df.to_dict(orient='records'):
    if row['dist'] == None:
        distances.append(route_to_distance[row['route']])
    else:
        distances.append(row['dist'])

df['dist'] = distances
```

Check data inconsistency in Air conditioner> # Hint group by count, but let them solve

```python
df.groupby('air conditioner').count()['date']
```

Fix in in air conditioner>

```python
df['air conditioner'] = df['air conditioner'].map({'offf': 'off',\
                                            'onn': 'on', 'oof': 'off', 'off': 'off', 'on': 'on'})
```

Check the data set for duplicates>

```python
df.duplicated().sum()
```

Remove duplicate rows>

```python
df = df.drop_duplicates()
# Two important arguments for this method are subset and keep
# These are used when we have a subset of columns that are eligible for identifying in object. For example a personal id, for people. 
# In the subset variable we can specify a list of column names, that will be considered when searching for duplicates.
# The keep argument will specifify which row to keep when a duplicate is found. Valid values are 'first' and 'last'
```

### **Data Conversion**

Convert the ordinal *trafic* column to a quantitative one>

```python
df['trafic'].map()
```

Label encode the *road* column using [Sklearn's LabelEncoder](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html)>

```python
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()

encoder.fit(df['road'])
df['road encoded'] = encoder.transform(df['road'])
df.head()
```

One Hot Encode the *road* column>

```python
pd.get_dummies(df['road'], prefix='road')
```

Normalize the *avg.cons.* column using the min-max scaling>

```python
df['avg.cons.'] = df['avg.cons.'].apply(lambda x: (x - df['avg.cons.'].min()) / 
                (df['avg.cons.'].max() - df['avg.cons.'].min()) )
```

Standardize the *avg.cons.* column>

```python
df['avg.cons.'] = df['avg.cons.'].apply(lambda x: (x - df['avg.cons.'].mean()) / 
                df['avg.cons.'].std() )
```

### **Dimensionality Reduction**

Visualize the correlation heatmap of the data and keep only the 2 most relevant predictors for *road* attribute>

```python
import seaborn as sns
import matplotlib as plt
from IPython.core.pylabtools import figsize
figsize(10, 10)

sns.heatmap(df.corr(), annot = True)

df_filtered = df[['road', 'dist', 'speed']]
```


Use [PCA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html) to create a new Dataframe of only 3 columns and the *road* attribute>

*Note: for this excersise I assume we have fixed all the data quality issues with. No missing values! If you haven't done yet, just load load in the fuel_data.txt that has no errors as the `df` variable.*

```python
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# converting the duration to number format, so the algorithm can work with it.
df['duration_minutes'] = df['duration'].apply(lambda time: int(time.split(':')[0]) * 60 + int(time.split(':')[1]))

# selecting the predictors into X
X = df[['duration_minutes', 'starttemp', 'endtemp', 'dist', 'num. persons', 'speed', 'avg.cons.', ]].values
# Scaling the variables 
X = StandardScaler().fit_transform(X)

pca = PCA(n_components=2)
principal_components = pca.fit_transform(X)
principal_df = pd.DataFrame(data = principal_components, columns = ['Principal Component 1', 'Principal Component 2'])

# Adding the target variable to the dataframe
principal_df['road'] = df['road']
principal_df
```

### **Excercise**


1. Visualize the difference between the effectiveness of PCA and Attribute selection! Hint: Use a scatterplot to visualize how well are both methods seperating the *road* target attribute.

1. Set the endtemp as the starttemp where *starttemp* is missing!

2. Fill the missing values for the *dist* by getting the median value for the same *route*! 

3. Fill all the remaining missing values!

3. We can see there are some outlier values in the *endtemp* column if we simply call `describe` method or plot the column's distribution. Do both and then come up with a solution on fixing these values!

4. Convert the *duration* column into *short*, *medium*, *long* values. Come up with the boundaries by splitting the duration range in 3 equal size ranges!

5. Standardize each column and create the Parallel Coordinates visualization using pandas's [built in method](https://pandas.pydata.org/docs/reference/api/pandas.plotting.parallel_coordinates.html)!