# EXNO:4-DS
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

```
Developed By: Sameer Shariff M
Register No.: 212224220085
```


# ALGORITHM:
STEP 1:Read the given Data.

STEP 2:Clean the Data Set using Data Cleaning Process.

STEP 3:Apply Feature Scaling for the feature in the data set.

STEP 4:Apply Feature Selection for the feature in the data set.

STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:
1.Filter Method
2.Wrapper Method
3.Embedded Method



# CODING AND OUTPUT:
```
import pandas as pd
from scipy import stats
import numpy as np
df=pd.read_csv("bmi.csv")
df.head()
```

**Output:**

<img width="571" height="368" alt="image" src="https://github.com/user-attachments/assets/33327449-539a-4ad7-a684-4a3b6fa02b9f" />

```
df_null_sum=df.isnull().sum()
df_null_sum
```

**Output:**

<img width="594" height="198" alt="image" src="https://github.com/user-attachments/assets/2124f62e-69e9-46cb-9158-1bf9fec4106a" />

```
 df.dropna()
 ```

**Output:**

<img width="578" height="521" alt="image" src="https://github.com/user-attachments/assets/6fbee574-75ae-4376-aa83-a2b2b97c0a8b" />

```
max_vals = np.max(np.abs(df[['Height', 'Weight']]), axis=0)
max_vals
```

**Output:**

<img width="836" height="145" alt="image" src="https://github.com/user-attachments/assets/9e671824-1190-41f0-a11d-b7b71a192b2e" />

```
from sklearn.preprocessing import StandardScaler
df1=pd.read_csv("bmi.csv")
df1.head()
```

**Output:**

<img width="721" height="313" alt="image" src="https://github.com/user-attachments/assets/402f9857-2659-4d9a-8589-fb080313723e" />

```
df1[['Height','Weight']]=sc.fit_transform(df1[['Height','Weight']])
df1.head(10)
```

**Output:**

<img width="847" height="467" alt="image" src="https://github.com/user-attachments/assets/6de6c237-0259-483c-bfd5-ef05779c6d8e" />

```
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
df.head(10)
```

**Output:**

<img width="1011" height="524" alt="image" src="https://github.com/user-attachments/assets/7f225a67-c3ad-46b3-8206-7a5cb955e7f7" />

```
from sklearn.preprocessing import MaxAbsScaler
scaler = MaxAbsScaler()
df3=pd.read_csv("bmi.csv")
df3.head()
df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
df
```

**Output:**

<img width="1019" height="653" alt="image" src="https://github.com/user-attachments/assets/f5efdc4d-2b61-4c44-b42c-4e17076f3f1f" />

```
from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()
df3[['Height','Weight']]=scaler.fit_transform(df3[['Height','Weight']])
df3.head()
```

**Output:**

<img width="1005" height="338" alt="image" src="https://github.com/user-attachments/assets/070de8b2-39a0-4872-b096-d8525b07b4ec" />

```
df=pd.read_csv("income(1) (1).csv")
df.info()
```

**Output:**

<img width="823" height="547" alt="image" src="https://github.com/user-attachments/assets/3321b085-1489-4b0d-a6bd-e59521240fa5" />

```
df_null_sum=df.isnull().sum()
df_null_sum
```

**Output:**

<img width="471" height="421" alt="image" src="https://github.com/user-attachments/assets/0e61229a-a7f9-441d-abb3-5b79b72f5917" />

```
categorical_columns = ['JobType', 'EdType', 'maritalstatus', 'occupation', 'relationship', 'race', 'gender', 'nativecountry']
df[categorical_columns] = df[categorical_columns].astype('category')
df[categorical_columns]
```

**Output:**

<img width="1503" height="569" alt="image" src="https://github.com/user-attachments/assets/db0d1631-305f-432b-90c1-3eb257e9926e" />

```
df[categorical_columns] = df[categorical_columns].astype('category')
df[categorical_columns] = df[categorical_columns].apply(lambda x: x.cat.codes)
##This code replaces each categorical column in the DataFrame with numbers that represent the categories.
df[categorical_columns]
```

**Output:**

<img width="1316" height="612" alt="image" src="https://github.com/user-attachments/assets/9a2c853c-3c3d-44f9-a2ee-b3b3d4be0ebc" />

```
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
```

**Output:**

<img width="1256" height="256" alt="image" src="https://github.com/user-attachments/assets/85815368-2160-4866-9f86-d84bf4c22d55" />

```
df=pd.read_csv("income(1) (1).csv")
df.info()
```

**Output:**

<img width="708" height="547" alt="image" src="https://github.com/user-attachments/assets/d4ed8ed3-9bda-47c7-a359-a8d40dcfc1da" />

```
import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2, f_classif
categorical_columns = ['JobType', 'EdType', 'maritalstatus', 'occupation', 'relationship', 'race', 'gender', 'nativecountry']
df[categorical_columns] = df[categorical_columns].astype('category')
df[categorical_columns]
```

**Output:**

<img width="1460" height="617" alt="image" src="https://github.com/user-attachments/assets/b47afae0-69cc-476d-8d1b-1106fb55421b" />

```
df[categorical_columns] = df[categorical_columns].apply(lambda x: x.cat.codes)
df[categorical_columns]
```

**Output:**

<img width="1064" height="547" alt="image" src="https://github.com/user-attachments/assets/8f5d038f-743d-4501-a7ec-c19322b1d2be" />

```
X = df.drop(columns=['SalStat'])
y = df['SalStat']
k_chi2 = 6
selector_chi2 = SelectKBest(score_func=chi2, k=k_chi2)
X_chi2 = selector_chi2.fit_transform(X, y)
selected_features_chi2 = X.columns[selector_chi2.get_support()]
print("Selected features using chi-square test:")
print(selected_features_chi2)
```

**Output:**

<img width="1012" height="307" alt="image" src="https://github.com/user-attachments/assets/1395f59d-4dcd-455f-9ec9-b73ea3d7a224" />

```
import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2, f_classif
from sklearn.model_selection import train_test_split # Importing the missing function
from sklearn.ensemble import RandomForestClassifier
selected_features = ['age', 'maritalstatus', 'relationship', 'capitalgain', 'capitalloss','hoursperweek']
X = df[selected_features]
y = df['SalStat']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
```

**Output:**

<img width="1352" height="358" alt="image" src="https://github.com/user-attachments/assets/36f6727e-7a0b-4c37-9247-2601da3fea45" />

```
y_pred = rf.predict(X_test)
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy using selected features: {accuracy}")
```

**Output:**

<img width="806" height="140" alt="image" src="https://github.com/user-attachments/assets/7512c3f7-cea7-4015-b211-9e4368f3a2d5" />

```
!pip install skfeature-chappers
```

**Output:**

<img width="1499" height="489" alt="image" src="https://github.com/user-attachments/assets/37ea9f08-bad6-4ca0-993b-2dcf38386eaa" />

```
df[categorical_columns] = df[categorical_columns].apply(lambda x: x.cat.codes)
df[categorical_columns]
```

**Output:**

<img width="1199" height="572" alt="image" src="https://github.com/user-attachments/assets/d32ad3fb-4e67-4b51-ab5d-6a3c4c8e7459" />

```
selected_features_anova = X.columns[selector_anova.get_support()]
print("\nSelected features using ANOVA:")
print(selected_features_anova)
```

**Output:**

<img width="1146" height="173" alt="image" src="https://github.com/user-attachments/assets/dece4d15-c2d5-4201-9788-03fb52eab6af" />

```
df[categorical_columns]
```

**Output:**

<img width="1016" height="537" alt="image" src="https://github.com/user-attachments/assets/8ee7fcbb-da4d-402b-ab30-c14423379a6e" />

```
rfe = RFE(estimator=logreg, n_features_to_select=n_features_to_select)
rfe.fit(X, y)
```

**Output:**

<img width="691" height="772" alt="image" src="https://github.com/user-attachments/assets/bab9c766-426d-47dc-80ad-24edb2b95206" />

# RESULT:

Thus the program to read the given data and perform Feature Scaling and Feature Selection process and
save the data to a file is been executed.
