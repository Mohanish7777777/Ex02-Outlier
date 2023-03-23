# Ex02-Outlier

You are given bhp.csv which contains property prices in the city of banglore, India. You need to examine price_per_sqft column and do following,

(1) Remove outliers using IQR 

(2) After removing outliers in step 1, you get a new dataframe.

(3) use zscore of 3 to remove outliers. This is quite similar to IQR and you will get exact same result

(4) for the data set height_weight.csv find the following

    (i) Using IQR detect weight outliers and print them

    (ii) Using IQR, detect height outliers and print them

# Explanatiom

An Outlier is an observation in a given dataset that lies far from the rest of the observations. That means an outlier is vastly larger or smaller than the remaining values in the set. An outlier is an observation of a data point that lies an abnormal distance from other values in a given population. (odd man out).Outliers badly affect mean and standard deviation of the dataset. These may statistically give erroneous results.Most machine learning algorithms do not work well in the presence of outlier. So it is desirable to detect and remove outliers.Outliers are highly useful in anomaly detection like fraud detection where the fraud transactions are very different from normal transactions.

# ALGORITHM

## STEP 1

Read the given Data.

## STEP 2

 Get the information about the data.

## STEP 3

Detect the Outliers using IQR method and Z score.

## STEP 4

Remove the outliers.

## STEP 5

Plot the datas using Box Plot.

# PROGRAM

Developed by : Mohanish.K
Registration Number : 22002294

```

import pandas as ps
import numpy as np
import seaborn as sns

df=ps.read_csv("bhp.csv")
df

df.head()
df.describe()
df.info()
df.isnull().sum()
df.shape

sns.boxplot(x="price_per_sqft",data=df)

q1=df['price_per_sqft'].quantile(0.35)
q3=df['price_per_sqft'].quantile(0.65)
print("First Quantile =",q1,"Second quantile =",q3)

IQR=q3-q1 #INTERQUARTILE RANGE
ul =q3+0.5*IQR
ll =q1-1.5*IQR

df1=df[((df['price_per_sqft']<=l1)&(df['price_per_sqft']>u1))]
df1

df1.shape

sns.boxplot(x='price_per_sqft',data=df1)

from scipy import stats
z=np.abs(stats.zscore(df['price_per_sqft']))
df2=df[(z<3)]
df2

print(df2.shape)

sns.boxplot(x='price_per_sqft',data=df2)

df3=ps.read_csv('height_weight.csv')
df3

df3.head()
df3.info()
df3.describe()
df3.isnull().sum()
df3.shape

sns.boxplot(x='weight',data=df3)

q1=df3['weight'].quantile(0.25)
q3=df3['weight'].quantile(0.75)
print('First Quantile =',q1,'Second Quantile =',q3)

IQR=q3-q1
u1=q3+1.5*IQR
l1=q1-1.5*IQR

df4 =df3[((df3['height']>=l1)&(df3['height']<=u1))]
df4

df4.shape

sns.boxplot(x='height',data=df4)
```

# OUTPUT

 DATASET FOR BHP_CSV

![2 1](https://user-images.githubusercontent.com/120620842/226962567-0d0d41a2-ff45-4c31-ad1f-bde333957133.png)

 DATASET HEAD(BHP)

![2 2](https://user-images.githubusercontent.com/120620842/226962873-3fa6db44-bd8c-470b-a9a6-0dd54c944503.png)

 DATASET DESCRIBE(BHP)

![2 3](https://user-images.githubusercontent.com/120620842/226963277-ff6a0634-1a9e-4665-af5a-f35b0d8b85ea.png)

 DATASET INFO(BHP)

![2 4](https://user-images.githubusercontent.com/120620842/226963692-d805f64d-a216-4ab9-9f1e-1a84a1279e9c.png)

 DATASET NULL VALUES(BHP)

![2 5](https://user-images.githubusercontent.com/120620842/226964172-a06974fb-8660-4cb6-b1b4-6e822446375d.png)

 DATASET SHAPE WITH OUTLIERS(BHP)

![2 6](https://user-images.githubusercontent.com/120620842/226964658-a286a0b5-c582-4729-b975-76892d559648.png)

 DATASET BOXPLOT WITH OUTLIERS(BHP)

![2 7](https://user-images.githubusercontent.com/120620842/226964941-fe6c110c-d9cd-4c05-8327-d21e117d6760.png)

 DATASET WITHOUT OUTLIERS(BHP)

![2 8](https://user-images.githubusercontent.com/120620842/226965517-4210efe0-1cfa-4e96-ad2a-190d60477c3a.png)

![2 9](https://user-images.githubusercontent.com/120620842/226966200-f16b3b8b-bdd9-40ec-88ac-19b44b32fc99.png)

 DATASET SHAPE WITHOUT OUTLIERS(BHP)

![2 10](https://user-images.githubusercontent.com/120620842/226966612-a256d840-d6b3-4575-b863-14220e928e05.png)

 DATASET BOXPLOT WITHOUT OUTLIERS(BHP)

![2 11](https://user-images.githubusercontent.com/120620842/226967376-7d6a8eaf-d2c6-4a25-8627-4e3ab046587d.png)

 DATASET AFTER REMOVAL OF OUTLIERS USING Z-SCORE(BHP)

![2 12](https://user-images.githubusercontent.com/120620842/226968071-482f12f8-d22c-4884-a73c-a8663e82eb5b.png)

 DATASET SHAPE AFTER REMOVAL OF OUTLIERS USING Z-SCORE(BHP)

![2 13](https://user-images.githubusercontent.com/120620842/226968605-9798e056-bec7-4309-87de-5416b4c80de7.png)

 DATASET BOXPLOT AFTER REMOVAL OF OUTLIERS USING Z-SCORE(BHP)

![2 14](https://user-images.githubusercontent.com/120620842/226969598-afd234e9-e162-4ed2-b132-22b1c4bf8c10.png)

 DATASET FOR WEIGHT_HEIGHT_CSV

![2 15](https://user-images.githubusercontent.com/120620842/226969918-2f0e7455-d5fa-43cc-b290-d726d417010a.png)

 DATASET HEAD(WEIGHT_HEIGHT)

![2 16](https://user-images.githubusercontent.com/120620842/226970344-29d41e84-8361-4852-9a5f-84c47e50d0cc.png)

 DATASET INFO(WEIGHT_HEIGHT)

![2 17](https://user-images.githubusercontent.com/120620842/226970686-3a2e61a6-9ebc-49ca-9c81-08608efb0956.png)

 DATASET DESCRIBE(WEIGHT_HEIGHT)

![2 18](https://user-images.githubusercontent.com/120620842/226971245-cdbef03a-144d-4909-af66-a008fdae0c86.png)

 DATASET NULL VALUES(WEIGHT_HEIGHT)

![2 19](https://user-images.githubusercontent.com/120620842/226971659-18678d44-0fe3-46bd-ae8e-fe082c5c7e9d.png)

 DATASET BOXPLOT WITH OUTLIERS(WEIGHT_HEIGHT)

![2 20](https://user-images.githubusercontent.com/120620842/226972009-65ece2b6-3914-4b29-a359-de2291ed1fe4.png)

 DATASET AFTER REMOVING OUTLIERS USING IQR METHOD(WEIGHT_HEIGHT)

![2 21](https://user-images.githubusercontent.com/120620842/226972573-9ea71781-9649-4227-ae17-d7e3d5f08fa0.png)

![image](https://user-images.githubusercontent.com/120620842/226973579-89b3e5b7-8bf1-447f-ac41-0175c879fe8f.png)

 DATASET SHAPE(WEIGHT_HEIGHT)

![image](https://user-images.githubusercontent.com/120620842/226973759-9f96fecb-b5f3-4531-8049-211fea7f1408.png)

 DATASET BOXPLOT AFTER REMOVING OUTLIERS USING IQR METHOD(WEIGHT_HEIGHT)

![image](https://user-images.githubusercontent.com/120620842/226974026-07b9adc6-633c-4f69-867d-07b9048a71b4.png)

# Result

DATASET BOXPLOT AFTER REMOVING OUTLIERS USING IQR METHOD(WEIGHT_HEIGHT).
