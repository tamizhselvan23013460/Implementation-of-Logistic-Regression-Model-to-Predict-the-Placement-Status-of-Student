# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1. Import pandas, numpy, matplotlib, and load the Placement Data CSV.
2. Drop 'sl_no' and 'salary' columns, and check dataset info.
3. Convert categorical columns to category type and then encode them numerically.
4. Separate features (x) and target (y) from the dataset.
5. Split the data into training and testing sets.
6. Create a logistic regression model and train it on the training data.
7. Predict on the test data and evaluate accuracy and confusion matrix.
8. Predict the output for two new custom input samples.


## Program :
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

data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)
data1.head()

data1.isnull().sum()

data1.duplicated().sum()

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

x=data1.iloc[:,:-1]
x

y=data1["status"]
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver="liblinear")
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
print("Accuracy=",accuracy)

from sklearn.metrics import confusion_matrix
confusion_matrix=confusion_matrix(y_test,y_pred)
confusion_matrix

from sklearn.metrics import classification_report
classification_report=classification_report(y_test,y_pred)
print(classification_report)

lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])
```

## Output:

### Preview datasets :

![EX_5_OUTPUT_1](https://github.com/user-attachments/assets/54d9f9a5-9d11-49c5-a01a-189ac07b78e5)


### Drop Sl.no & Salary :

![EX_5_OUTPUT_2](https://github.com/user-attachments/assets/f598fc13-a904-4d77-81eb-25552ef78823)


### Find null Value :

![EX_5_OUTPUT_3](https://github.com/user-attachments/assets/71f3164b-f833-4232-b1af-a905a91eab17)


### Find Duplicate Value :
![EX_5_OUTPUT_4](https://github.com/user-attachments/assets/bfb2d297-205d-4c07-860d-602a336fef8b)

### Implement label Encoder :

![EX_5_OUTPUT_5](https://github.com/user-attachments/assets/4d44591f-d7f1-495e-b8bf-66d236cc749c)

### Initialize X value :

![EX_5_OUTPUT_6](https://github.com/user-attachments/assets/9ee120f6-ac1f-4cae-965c-8d419eea5198)

### Assign Y as Status :

![EX_5_OUTPUT_7](https://github.com/user-attachments/assets/03aafaff-ea53-4d99-8103-4f101ec4ac87)


### Y_Predict :
![EX_5_OUTPUT_8](https://github.com/user-attachments/assets/696ee833-e3aa-4d8f-bd69-591b0e988bec)


### Accuracy :
![EX_5_OUTPUT_9](https://github.com/user-attachments/assets/ea410d17-6d07-4a3d-bebc-af86c81b5623)


### Confusion Matrix :
![EX_5_OUTPUT_10](https://github.com/user-attachments/assets/9327aad4-bf0d-42e8-90e9-a347a8ec77a8)


### Classification Report :

![EX_5_OUTPUT_11](https://github.com/user-attachments/assets/108a6acb-4eda-45aa-b02a-7f1c0029ccb0)

### Final Predict :

![EX_5_OUTPUT_12](https://github.com/user-attachments/assets/48d38321-a0ac-472f-817b-1b9967065c93)



## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
