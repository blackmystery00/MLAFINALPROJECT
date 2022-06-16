import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import load_diabetes, load_boston
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import seaborn as sns

#---------------------------------#
# Page layout
## Page expands to full width
st.set_page_config(page_title='The Machine Learning App',
    layout='wide')

#---------------------------------#
# Model building

# function to add
# function to add
def add_parameter_ui(clf_name):
    params = dict()
    if clf_name == "KNN":
        K = st.sidebar.slider("K", 1, 15)
        params["K"] = K
    elif clf_name == "SVM":
        C = st.sidebar.slider("C", 0.01, 10.0)
        params["C"] = C
    else:
        max_depth = st.sidebar.slider("max_depth", 2, 15)
        n_estimators = st.sidebar.slider("n_estimators", 1, 100)
        params["max_depth"] = max_depth
        params["n_estimators"] = n_estimators
    return params


def get_classifier(clf_name, params):
    if clf_name == "KNN":
        clf = KNeighborsClassifier(n_neighbors=params["K"])
    elif clf_name == "SVM":
        clf = SVC(C=params["C"])
    else:
        clf = RandomForestClassifier(n_estimators=params["n_estimators"], 
        max_depth=params["max_depth"], random_state=1234)
    return clf
    
def build_model(df):
    
    original = df
    st.markdown('**1.2. Original Data Shape**')
    st.info(original.shape)
    
    st.markdown('**1.3. Example of original Data loc**')
    st.write(original.loc[original['PERINGKAT']=='RUNCIT'])
    
    st.markdown('**1.4. Example of original column 4 till 9 Data iloc**')
    st.write(original.iloc[:,[3,4,5,6,7,8]])
    
    st.subheader('2. Handling missing values')
    st.markdown('**2.1. Data before handling missing values**')
    st.write(original)
    
    st.markdown('**2.2. Deletion**')
    deletion = original.dropna()
    st.write(deletion)
    
    st.markdown('**2.2. Imputation (Mean)**')
    dataImpute2 = original.loc[:,:].fillna(value=original.loc[:,:].mean())
    dataImpute2 
    
    

    x = deletion.iloc[:,[3,4,5,6,7]] # Using all column except for the last column as X
    y = deletion.iloc[:,-1] # Selecting the last column as Y
    y=y.astype('int')
    st.write("number of classes", len(np.unique(y)))
    # Data splitting
    st.subheader('3. Data splitting')
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=(100-split_size)/100, random_state=0)

    st.markdown('**3.1. Data splits info**')
    st.write('Training set')
    st.info(x_train.shape)
    st.write('Test set')
    st.info(x_test.shape)

    st.markdown('**3.2. Variable details**:')
    st.write('X variable')
    st.info(list(x.columns))
    st.write('Y variable')
    st.info(y.name)


    sc_x=StandardScaler()
    x_train=sc_x.fit_transform(x_train)
    x_test=sc_x.fit_transform(x_test)
    if classifier_name == "KNN":
        classifier=KNeighborsClassifier(n_neighbors=5,metric ='minkowski',p=2) #using Euclidean distance
        classifier.fit(x_train,y_train) 
        y_pred = classifier.predict(x_test)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='macro')
        st.write(f"classifier = {classifier_name}")
        st.write(f"Accuracy = {acc}")
        st.write(f"Precision  = {prec}")
    else:
        classifier = SVC(kernel='linear', random_state=0)
        classifier.fit(x_train,y_train)
        y_pred=classifier.predict(x_test)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='macro')
        st.write(f"classifier = {classifier_name}")
        st.write(f"Accuracy = {acc}")
        st.write(f"Precision  = {prec}")
        
    





#---------------------------------#
st.write("""
# The Machine Learning App about average vegetable price using KNN or SVM

In this implementation, the KNN and SVM function is used in this app for build a regression model using either KNN or SVM algorithm.


""")

#---------------------------------#
# Sidebar - Collects user input features into dataframe
with st.sidebar.header('1. Upload your CSV data'):
    uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
    st.sidebar.markdown("""
[Example CSV input file](https://raw.githubusercontent.com/dataprofessor/data/master/delaney_solubility_with_descriptors.csv)
""")

# create classifier_name listbox
classifier_name = st.sidebar.selectbox("Select Classifier", ("KNN", "SVM"))

# Sidebar - Specify parameter settings
with st.sidebar.header('2. Set Parameters'):
    split_size = st.sidebar.slider('Data split ratio (% for Training Set)', 10, 90, 80, 5)

#---------------------------------#
# Main panel

# Displays the dataset
st.subheader('1. Dataset')

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.markdown('**1.1. Glimpse of the original dataset**')
    st.write(df)
    build_model(df)
else:
    st.info('Awaiting for CSV file to be uploaded.')
    if st.button('Press to use Example Dataset'):
        # Diabetes dataset
        #diabetes = load_diabetes()
        #X = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
        #Y = pd.Series(diabetes.target, name='response')
        #df = pd.concat( [X,Y], axis=1 )

        #st.markdown('The Diabetes dataset is used as the example.')
        #st.write(df.head(5))

        # Boston housing dataset
        boston = load_boston()
        X = pd.DataFrame(boston.data, columns=boston.feature_names)
        Y = pd.Series(boston.target, name='response')
        df = pd.concat( [X,Y], axis=1 )

        st.markdown('The Boston housing dataset is used as the example.')
        st.write(df.head(5))

        build_model(df)