# importing libraries
import streamlit as st
import numpy as np 
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

#Heading 
st.write('''Exploring different ML Models with different Datasetas
         Exploring..
         ''')

#creating a sidebar for different datasets with there names
dataset_name = st.sidebar.selectbox(
    'Select Dataset',
    ('Iris', 'Breast Cancer', 'Wine')
)

#creating a sidebar for different ML Models with there names
classifier_name = st.sidebar.selectbox(
    'Select classifier',
    ('KNN', 'SVM', 'Random Forest')
)


#creating a function to load datasets
def get_dataset(dataset_name):
    data= None
    if dataset_name == "Iris":
        data = datasets.load_iris()
    elif dataset_name == "Wine":
        data = datasets.load_wine()
    else:
        data = datasets.load_breast_cancer()
    x = data.data
    y = data.target
    return x, y 

#calling function and assigning it to x & y 
X, y = get_dataset(dataset_name)

#to display the shape of the dataset 
st.write('Shape of dataset:', X.shape)
st.write('number of classes: ', len(np.unique(y)))

#creating a function for choosing slider classifiers 
def add_parameter_ui(classifier_name):
    params = dict() #creating a empty dictionary
    if classifier_name == 'SVM':
        C = st.sidebar.slider('C', 0.01, 10.0)
        params['C'] = C #its the degree of correct classifiction
    elif classifier_name == 'KNN':
        K = st.sidebar.slider('K', 1, 15)
        params['K'] = K #its the degree of nearest neighbour
    else:
        max_depth = st.sidebar.slider('max_depth', 2, 15)
        params['max_depth'] = max_depth #depth of every tree that will be in Random Forest Regressor
        n_estimators = st.sidebar.slider('n_estimators', 1, 100)
        params['n_estimators'] = n_estimators #number of Trees
    return params

#calling the function
params = add_parameter_ui(classifier_name)
    
#creataing a function for the classifier

def get_classifier(classifier_name, params):
    clf = None 
    if classifier_name == 'SVM':
        clf = SVC(C=params['C'])
    elif classifier_name == 'KNN':
        clf = KNeighborsClassifier(n_neighbors=params['K'])
    else:
        clf = RandomForestClassifier(n_estimators=params['n_estimators'],
                max_depth=params['max_depth'], random_state=1234)
    return clf

#calling the function
clf = get_classifier(classifier_name, params)

#splitting the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

#training the modle 
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

#to check the models accuracy_score
acc = accuracy_score(y_test, y_pred)
st.write(f'Classifier : {classifier_name}')
st.write(f'Accuracy :', acc)

####Plotting Charts####
#ab hum apny sary sary features ko 2 dimentional plot pay draw kr dayn gay using pcs
pca = PCA(2)
X_projected = pca.fit_transform(X)

#ab hum apny sary data 0 or 1 dimenssion may sclice kar kar day gay 
x1 = X_projected[:, 0]
x2 = X_projected[:, 1]

fig = plt.figure()
plt.scatter(x1, x2,
            c=y, alpha=0.8,
            cmap='viridis')

plt.xlabel('Principal Components 1')
plt.ylabel('Principal Components 2')
plt.colorbar()

#plt.show()
st.pyplot(fig)










