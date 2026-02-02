import joblib
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np 
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import cross_validate

#"C:\Users\boris\Downloads\tester.xlsx"
data=pd.read_excel(r"C:\Users\boris\OneDrive\桌面\project_ques\knn部分\project.xlsx") #匯入檔案名
#print(data)
data1=[]
data2=[]
#for i in range(2,216):
#print(data[2:4])
train_data = np.array(data)
train_list=train_data.tolist()
#print(train_list)
for f in train_list:
    data1.append(f[1:15])#隨不同數量資料要改
    data2.append(f[15:])
    #print(f[1:16])
    #print(f[16:])
#print(data2)

#dx , dy = make_blobs(n_samples=100, n_features=2, centers=2, random_state=0) #帶入資料的地方
#dx_std = StandardScaler().fit_transform(dx)#標準化
#dx_train, dx_test, dy_train, dy_test = train_test_split(dx_std, dy, test_size=0.2, random_state=0) #設定測資

dx_train, dx_test, dy_train, dy_test = train_test_split(data1, data2, test_size=0.2, random_state=0) #設定測資(分成訓練組與測試組)


#knn = KNeighborsClassifier(n_neighbors=9)#訓練(看參數)



#knn.fit(dx_train, dy_train) #帶入訓練值

#predictions = knn.predict(dx_test) #藉由(實際測試組)這些特徵判斷是甚麼東西


#k_range = range(2,20)
k_scores = []
#for k in  range(2,20):
 
knn_model = KNeighborsClassifier(n_neighbors = 3) #k的值

knn_model.fit(data1, data2) #帶入訓練值


scoring = ['accuracy']

accuracy = cross_validate(knn_model, data1, data2, cv=5, scoring="accuracy",return_train_score = True)

k_scores.append(np.mean(accuracy['train_score']))
#print(k)
print("Train Accuracy Scores:", accuracy['train_score'],"Train Accuracy mean Scores:",np.mean(accuracy['train_score']))
print("Test Accuracy Scores:", accuracy['test_score'],"Test Accuracy mean Scores:",np.mean(accuracy['test_score']))
joblib.dump(knn_model, 'model.joblib')

 







 
 #print("cross validation:",accuracy) 

 #print("K="+ str(k) +" Accuracy= "+ str(accuracy.mean()))

 #k_scores.append(accuracy.mean())


 #predictions = knn_model.predict(dx_test)

 #print("實際值",dy_test)#跑出來的
 #print("電腦答案",predictions)#正確的

 #print("訓練率:",knn_model.score(data1, data2))#上面分的
 #print("答對率:",knn_model.score(dx_test, dy_test))


 

