# Extract-Transform-Load (ETL) in Python with Pandas

#### Install pandas

As with any libary pandas has to be installed first. In the command line issue>

```bash
pip install pandas
```


#### Pandas

Pandas is a data analysis and manipulation tool. [Link](https://pandas.pydata.org/)

#### Download data

We will be using the Fuel Dataset. [Download](https://raw.githubusercontent.com/VSZM/ELTE_Adatbanyaszat_es_Gepi_tanulas/master/practice/fuel_data.txt)


#### Import

Before using pandas we must import it>

```python
import pandas as pd
```

#### Read data

Read the previously downloaded fuel data>

```python
df = pd.read_csv('practice/fuel_data.txt', delimiter='\t')
```

We can read all kinds of data, like sql, json, excel as well. Check documentation for more details.

#### Pandas main types

*Series*: Series is a serial container. It can represent lists, columns of data, etc. 

*DataFrame*: DataFrame is a tabular container. It can represent SQL tables, Excel files, etc. 

#### Explore data

We can inspect the DataFrame contents with a lot of commands.

See first first rows of the DataFrame>

```python
df.head() # has a default parameter of 5 for first argument controlling the number of displayed rows
```

See the last rows of the DataFrame>

```python
df.tail() # has a default parameter of 5 for first argument controlling the number of displayed rows
```

Get a sample of the data>

```python
df.sample() # has a default parameter of 1 for first argument controlling the sample size
```


Get a description of the DataFrame. This will show statistics about the numerical data>

```python
df.describe()
```

Show the column names of the DataFrame>

```python
df.columns
```

Show the types for each column>

```python
df.dtypes
```

Show index of the DataFrame>

```python
df.index
```

#### Read and search


Get a single row by position>

```python
df.iloc[0]
```

Get a single row by indexer>

```python
df.loc[0]
```
*Note: Indexers are not necessarily numerical. Hence the difference between them.*

Get a single column as a Series>

```python
df['date']
```

Get a single cell value>

```python
df.at[0, 'date']
```

Get a single column as a new DataFrame>

```python
df[['date']]
```

Slice multiple columns as a new DataFrame>

```python
df[['date', 'starttime']]
```


Filter the rows of the DataFrame>

```python
df[df['num. persons'] == 1]
# Syntax explanation:
# Inside the first brackets (df[]) we put a list of boolean values. 
# We could simply put df[[True]*80 + [False]*20] to select the first 80 rows
# The expression inside (df['num. persons'] == 1) will return such a boolean list
# by checking each value of the 'num. persons' column for equality to 1
```

Filter based on string>

```python
df[df['route'].str.contains('bp')] 
# Here the .str will convert the values as strings so we can use string operations
```

Iterate rows of the DataFrame>

```python
for row in df.iterrows():
    print(row)
# Here we see that row will be a 2-tuple. The first part is the index of the row, the second part is a dictionary representing the row data where the keys are the column names.
```

#### Transformations, Aggregations, Groups


Get unique values of a column>

```python
df['fuel type'].unique()
```

Get number of unique items>

```python
df['fuel type'].nunique()
```

Built in aggregations>

```python
df['starttemp'].sum()
df['starttemp'].min()
df['starttemp'].max()
# etc
```

Transform a column>

```python
df['starttemp'].apply(lambda x: x * 3)
```

Grouping rows>

```python
df.groupby('road').count()
df.groupby('road')['starttemp'].min()
# multi level group by
df.groupby(['road', 'air conditioner'])['starttemp'].min()
```

#### Modifying the Data

Add row to the DataFrame>

```python
df = df.append({'date':'2019.02.18', 'starttime': '08:30', 'endtime':'09:00', 'starttemp': 12, 'endtemp': 13,'air conditioner': 'off', 'trafic':'low', 'route':'home-elte', 'dist': 30, 'avg.cons.': 0, 'speed':30, 'duration':'00:30', 'fuel type': None, 'road':'normal', 'num. persons':1}, ignore_index=True)
```

Add column to the DataFrame>

```python
df['enjoyed the ride'] = [True] * len(df)
```

Modify a cell's value>

```python
df.at[0, 'enjoyed the ride'] = False
```


Drop column from the DataFrame>

```python
df = df.drop('enjoyed the ride', axis = 1)
```



#### Exercise

1. Split the date to Year, Month, Date columns!

2. Show the travels of February!

3. Show the travels of the second half of the year!

1. What percent of all the travels where conducted in low traffic?

1. What was the maximal heat decrease during a trip?

3. What was the average distance of travels starting from Budapest?

3. What was the sum of length of travels going to Budapest? (in minutes)

1. Calculate the average fuel consumption for each of the fuel types! What difference do you see between having the air conditioner turned on vs off? 

3. Add a new column for cost efficiency by calculating the average consumption per person. 

3. Get the fastest driver for each route! (We assume the roads are symmetric thus bp-dujv is the same as dujv-bp)

3. Create 2 new columns 'Origin' and 'Destination' and store the capitalized values extracted from 'route' column!

#### Further info, learning

[Cheatsheet](https://pandas.pydata.org/Pandas_Cheat_Sheet.pdf)

[A not so 10 minute tutorial](https://pandas.pydata.org/pandas-docs/stable/getting_started/10min.html#min)