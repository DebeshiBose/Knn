import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv(r'C:\Users\Debeshi\Downloads\winequality.csv - winequality.csv.csv')
print(df)
plt.scatter(df['quality'], df['fixed acidity'], color = 'maroon')
plt.title('relation of fixed acidity with wine')
plt.xlabel('quality')
plt.ylabel('fixed acidity')
plt.legend()
plt.show()
sns.countplot(df['quality'])
plt.show()
qt = (3, 4, 5)
group_names = ['bad', 'good']
df['quality'] = pd.cut(df['quality'], bins = qt, labels = group_names)
sns.countplot(df['quality'])
plt.show()
plt.figure(figsize=(9,3))
sns.histplot(df,x='quality', y='type', hue='quality')
plt.show()
plt.figure(figsize=[19,10],facecolor='white')
sns.heatmap(df.corr(),annot=True)
plt.show()
from sklearn.model_selection import train_test_split

X=df[['fixed acidity','volatile acidity','citric acid','residual sugar', 'chlorides','free sulfur dioxide','total sulfur dioxide','density','pH','sulphates', 'alcohol']]# independenct
Y=df['quality']# dependent
X_train,X_test,y_train,y_test=train_test_split(X,Y,test_size = .10,random_state=0)

from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=1)

knn.fit(X_train,y_train)

#let us get the predictions using the classifier we had fit above
y_pred = knn.predict(X_test)
print("pred",y_pred)
print("real",y_test.values)

from sklearn.metrics import accuracy_score
print("Accuracy is",accuracy_score(y_test, y_pred))
from sklearn.metrics import confusion_matrix
confusionMatrix = confusion_matrix (y_test, y_pred)
confusionMatrix