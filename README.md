# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

Step 1. Start

Step 2. Load the California Housing dataset and select the first 3 features as input (X) and target variables (Y) (including the target price and another feature).

Step 3. Split the data into training and testing sets, then scale (standardize) both the input features and target variables.

Step 4. Train a multi-output regression model using Stochastic Gradient Descent (SGD) on the training data.

Step 5. Make predictions on the test data, inverse transform the predictions, calculate the Mean Squared Error, and print the results.

Step 6. Stop


## Program & Output:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: TAMIZHSELVAN B
RegisterNumber:  212223230225
*/
```
```
import pandas as pd
data=pd.read_csv('Placement_data.csv')
data.head(5)
```

![EX_5_OUTPUT_1](https://github.com/user-attachments/assets/54d9f9a5-9d11-49c5-a01a-189ac07b78e5)


```
data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)
data1.head()
```

![EX_5_OUTPUT_2](https://github.com/user-attachments/assets/f598fc13-a904-4d77-81eb-25552ef78823)


```
data1.isnull().sum()
```

![EX_5_OUTPUT_3](https://github.com/user-attachments/assets/71f3164b-f833-4232-b1af-a905a91eab17)


```
data1.duplicated().sum()
```

![EX_5_OUTPUT_4](https://github.com/user-attachments/assets/bfb2d297-205d-4c07-860d-602a336fef8b)



```
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"])
data1["status"]=le.fit_transform(data1["status"])
data1
```


![EX_5_OUTPUT_5](https://github.com/user-attachments/assets/4d44591f-d7f1-495e-b8bf-66d236cc749c)


```
x=data1.iloc[:,:-1]
x
```


![EX_5_OUTPUT_6](https://github.com/user-attachments/assets/9ee120f6-ac1f-4cae-965c-8d419eea5198)



```
y=data1["status"]
y
```


![EX_5_OUTPUT_7](https://github.com/user-attachments/assets/03aafaff-ea53-4d99-8103-4f101ec4ac87)


```
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
```
```
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver="liblinear")
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
y_pred
```

![EX_5_OUTPUT_8](https://github.com/user-attachments/assets/696ee833-e3aa-4d8f-bd69-591b0e988bec)


```
from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
print("Accuracy=",accuracy)
```

![EX_5_OUTPUT_9](https://github.com/user-attachments/assets/ea410d17-6d07-4a3d-bebc-af86c81b5623)


```
from sklearn.metrics import confusion_matrix
confusion_matrix=confusion_matrix(y_test,y_pred)
confusion_matrix
```

![EX_5_OUTPUT_10](https://github.com/user-attachments/assets/9327aad4-bf0d-42e8-90e9-a347a8ec77a8)


```
from sklearn.metrics import classification_report
classification_report=classification_report(y_test,y_pred)
print(classification_report)
```

![EX_5_OUTPUT_11](https://github.com/user-attachments/assets/108a6acb-4eda-45aa-b02a-7f1c0029ccb0)




## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
