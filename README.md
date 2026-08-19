# EXNO:4-DS
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

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
 ~~~
import pandas as pd 
from scipy import stats 
import numpy as np
df=pd.read_csv("bmi.csv") 
df.head()
~~~
<img width="422" height="202" alt="Screenshot 2026-08-19 222006" src="https://github.com/user-attachments/assets/c5ed8ba1-8d4e-4635-811e-dee1ea463fb9" />

~~~
df_null_sum=df.isnull().sum() 
df_null_sum
~~~
<img width="271" height="127" alt="Screenshot 2026-08-19 222123" src="https://github.com/user-attachments/assets/2e6781c0-c088-48cc-a2fc-71acbd1da958" />

~~~
df.dropna()
~~~
<img width="530" height="442" alt="Screenshot 2026-08-19 222301" src="https://github.com/user-attachments/assets/c4ab4dd7-cb3a-40e7-8a95-67e7f95cc2f9" />

~~~
max_vals = np.max(np.abs(df[['Height', 'Weight']]), axis=0) 
max_vals
~~~
<img width="252" height="82" alt="Screenshot 2026-08-19 222438" src="https://github.com/user-attachments/assets/45890451-7524-42ab-b091-dd43a5e2bd92" />

~~~
from sklearn.preprocessing import StandardScaler 
df1=pd.read_csv("bmi.csv") 
df1.head()
~~~
<img width="483" height="205" alt="Screenshot 2026-08-19 222541" src="https://github.com/user-attachments/assets/a0d9e9c0-c1e4-4940-a70e-720b49bcbaf1" />

~~~
sc=StandardScaler()
df1[['Height','Weight']]=sc.fit_transform(df1[['Height','Weight']]) 
df1.head(10)
~~~
<img width="517" height="366" alt="Screenshot 2026-08-19 222644" src="https://github.com/user-attachments/assets/108ed488-cd58-443e-a11d-1f6deb9289fb" />

~~~
from sklearn.preprocessing import MinMaxScaler 
scaler=MinMaxScaler() 
df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
df.head(10)
~~~
<img width="527" height="360" alt="Screenshot 2026-08-19 222759" src="https://github.com/user-attachments/assets/dd2f46d4-faee-4438-a7eb-b97ac3839938" />

~~~
from sklearn.preprocessing import MaxAbsScaler 
scaler = MaxAbsScaler() 
df3=pd.read_csv("bmi.csv") 
df3.head()
df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']]) 
df
~~~
<img width="477" height="208" alt="Screenshot 2026-08-19 222858" src="https://github.com/user-attachments/assets/595ebb0f-051d-46ba-a380-8a450c49b4ac" />

~~~
df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
df
~~~
<img width="522" height="437" alt="Screenshot 2026-08-19 223012" src="https://github.com/user-attachments/assets/b3a50ee3-8022-4027-b514-510e7531d038" />

~~~
from sklearn.preprocessing import RobustScaler 
scaler = RobustScaler() 
df3[['Height','Weight']]=scaler.fit_transform(df3[['Height','Weight']]) 
df3.head()
~~~
<img width="522" height="220" alt="Screenshot 2026-08-19 223055" src="https://github.com/user-attachments/assets/86a80b47-01a5-4ce8-b89a-9ba55371ecdd" />

~~~
df=pd.read_csv("income(1) (1).csv") 
df.info()
~~~
<img width="506" height="445" alt="Screenshot 2026-08-19 223137" src="https://github.com/user-attachments/assets/cab3d1d8-8346-4182-9fc3-f6709465bde7" />

~~~
df_null_sum=df.isnull().sum() 
df_null_sum
~~~
<img width="447" height="312" alt="Screenshot 2026-08-19 223228" src="https://github.com/user-attachments/assets/87b79df5-6ad5-43fc-a188-a6e32305d7f8" />

~~~
categorical_columns = ['JobType', 'EdType', 'maritalstatus', 'occupation', 'relationship', 'race', 'gender', 'nativecountry'] 
df[categorical_columns] = df[categorical_columns].astype('category')
df[categorical_columns]
~~~
<img width="1022" height="438" alt="Screenshot 2026-08-19 223314" src="https://github.com/user-attachments/assets/65c43707-19ce-4abd-bfe9-002924f6988b" />

~~~
df[categorical_columns] = df[categorical_columns].astype('category') 
df[categorical_columns] = df[categorical_columns].apply(lambda x: x.cat.codes) 
df[categorical_columns]
~~~
<img width="911" height="452" alt="Screenshot 2026-08-19 223404" src="https://github.com/user-attachments/assets/0e30d933-544b-4a76-bd59-87593b9097d3" />

~~~
X = df.drop(columns=['SalStat']) 
y = df['SalStat']
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score 
from sklearn.ensemble import RandomForestClassifier 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
rf = RandomForestClassifier(n_estimators=100, random_state=42) 
rf.fit(X_train, y_train)
~~~
<img width="475" height="121" alt="Screenshot 2026-08-19 223507" src="https://github.com/user-attachments/assets/bdcafaa3-f93f-419f-9ce3-0e5da4a2270b" />

~~~
y_pred = rf.predict(X_test)
df=pd.read_csv("income(1) (1).csv") 
df.info()
~~~
<img width="442" height="442" alt="Screenshot 2026-08-19 223629" src="https://github.com/user-attachments/assets/8910a0b1-751e-41ba-b249-a12b9ab95b79" />

~~~
import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2, f_classif 
categorical_columns = ['JobType', 'EdType', 'maritalstatus', 'occupation', 'relationship', 'race', 'gender', 'nativecountry'] 
df[categorical_columns] = df[categorical_columns].astype('category')
df[categorical_columns]
~~~
<img width="1042" height="441" alt="Screenshot 2026-08-19 223719" src="https://github.com/user-attachments/assets/153cf76c-f943-4f69-904a-707fe2484320" />

~~~
df[categorical_columns] = df[categorical_columns].apply(lambda x: x.cat.codes)
df[categorical_columns]
~~~
<img width="877" height="447" alt="Screenshot 2026-08-19 223814" src="https://github.com/user-attachments/assets/9173ea13-26e8-4e46-a22b-44ea7c59651a" />

~~~
X = df.drop(columns=['SalStat']) 
y = df['SalStat'] 
k_chi2 = 6
selector_chi2 = SelectKBest(score_func=chi2, k=k_chi2) 
X_chi2 = selector_chi2.fit_transform(X, y) 
selected_features_chi2 = X.columns[selector_chi2.get_support()] 
print("Selected features using chi-square test:") 
print(selected_features_chi2) 
~~~
<img width="768" height="107" alt="Screenshot 2026-08-19 223907" src="https://github.com/user-attachments/assets/d2c9d520-804d-4cbf-8df8-5d7b793c49da" />

~~~
import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2, f_classif 
from sklearn.model_selection import train_test_split 
from sklearn.ensemble import RandomForestClassifier
selected_features = ['age', 'maritalstatus', 'relationship', 'capitalgain', 'capitalloss', 'hoursperweek'] 
X = df[selected_features] 
y = df['SalStat'] 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42) 
rf = RandomForestClassifier(n_estimators=100, random_state=42) 
rf.fit(X_train, y_train)
~~~
<img width="447" height="112" alt="Screenshot 2026-08-19 223959" src="https://github.com/user-attachments/assets/9e2bc520-f765-4807-87e2-86834c6f5178" />

~~~
y_pred = rf.predict(X_test) 
from sklearn.metrics import accuracy_score 
accuracy = accuracy_score(y_test, y_pred) 
print(f"Model accuracy using selected features: {accuracy}")
~~~
<img width="642" height="42" alt="Screenshot 2026-08-19 224045" src="https://github.com/user-attachments/assets/68f4fe78-437c-4428-bd3f-f88b2d52f0e3" />

~~~
!pip install skfeature-chappers
~~~
<img width="1247" height="486" alt="Screenshot 2026-08-19 224126" src="https://github.com/user-attachments/assets/bc9d5ff5-e90e-4a78-856d-c6f364c74aca" />

~~~
import numpy as np 
import pandas as pd 
from skfeature.function.similarity_based import fisher_score
from sklearn.ensemble import RandomForestClassifier 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score
categorical_columns = [ 'JobType', 'EdType', 'maritalstatus', 'occupation', 'relationship', 'race', 'gender', 'nativecountry' ]
df[categorical_columns] = df[categorical_columns].astype('category')
df[categorical_columns]
~~~
<img width="893" height="445" alt="Screenshot 2026-08-19 224229" src="https://github.com/user-attachments/assets/31005dff-13f0-4d18-9fe1-66c92143ba99" />

~~~
X = df.drop(columns=['SalStat']) 
y = df['SalStat']
k_anova = 5 
selector_anova = SelectKBest(score_func=f_classif,k=k_anova) 
X_anova = selector_anova.fit_transform(X, y)
selected_features_anova = X.columns[selector_anova.get_support()]
print("\nSelected features using ANOVA:")
print(selected_features_anova)
~~~
<img width="915" height="85" alt="Screenshot 2026-08-19 224317" src="https://github.com/user-attachments/assets/370c700f-ad91-4e87-bad5-52b31cd1465b" />

~~~
import pandas as pd 
from sklearn.feature_selection import RFE 
from sklearn.linear_model import LogisticRegression 
df=pd.read_csv("income(1) (1).csv")
categorical_columns = [ 'JobType', 'EdType', 'maritalstatus', 'occupation', 'relationship', 'race', 'gender', 'nativecountry' ]
df[categorical_columns] = df[categorical_columns].astype('category')
df[categorical_columns] = df[categorical_columns].apply(lambda x: x.cat.codes)
df[categorical_columns]
~~~
<img width="936" height="447" alt="Screenshot 2026-08-19 224359" src="https://github.com/user-attachments/assets/3710f944-4196-4ff5-96d3-7c51f8769345" />

~~~
X = df.drop(columns=['SalStat']) 
y = df['SalStat'] 
logreg = LogisticRegression()
n_features_to_select =6
rfe = RFE(estimator=logreg, n_features_to_select=n_features_to_select) 
rfe.fit(X, y)
~~~
<img width="1026" height="753" alt="Screenshot 2026-08-19 224455" src="https://github.com/user-attachments/assets/d17187a6-009a-4d2f-b502-f9e28a175714" />


# RESULT:
       The given dataset was successfully read, and Feature Scaling and Feature Selection were performed. The selected and scaled features were saved successfully into a new data file.
