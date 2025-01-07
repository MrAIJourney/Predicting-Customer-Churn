import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# import graphviz
from sklearn.preprocessing import StandardScaler,OneHotEncoder,LabelEncoder,MinMaxScaler
from sklearn.compose import ColumnTransformer
# from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.model_selection import train_test_split,GridSearchCV,cross_val_score,KFold,RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.datasets import make_classification
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.metrics import classification_report,confusion_matrix ,roc_auc_score,ConfusionMatrixDisplay,accuracy_score,precision_score,recall_score,f1_score,precision_recall_curve
from sklearn.metrics import mean_squared_error
from sklearn import tree
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
import sqlite3
# import folium
# from folium.plugins import MarkerCluster
from sklearn.neighbors import KNeighborsClassifier
# from plotly.graph_objs import *
# import plotly.express as px 
# import plotly.graph_objects as go
# from plotly.subplots import make_subplots
from sklearn.pipeline import Pipeline
# from hyperopt import fmin, tpe, hp, Trials
import warnings
# from IPython.display import Image, display,IFrame
# from plotly.offline import plot
# from xgboost import XGBClassifier




# Suppress all warnings
warnings.filterwarnings('ignore')

# <---- load the data: ---->
data = pd.read_csv('customer_churn_dataset-testing-master.csv')
# print(data.head(25))

warnings.filterwarnings('ignore', category=UserWarning, message='find font: font family')

# <---- know about shape and type of dataset ---->
# print('Shape of dataset ---->  ', np.shape(data))
# print('Size of dataset ---->  ', np.size(data))
# print('Types of data in dataset ---->  ', data.dtypes)

# <---- know number of each item in each column ---->
# for col in data.columns:
#     print(f'Count of item in column {col}\n', data[col].value_counts())
#     print('*'*25)

# <---- know the unique value in each column ---->
# for col in data.columns:
#     print(f'Unique values in col {col}:\n{data[col].unique()}')
#     print('*'*25)

# <--- Cleaning data --->
# check for null values
# print(data.isna().mean())

# drop 'Customer ID' column
customer_id = data['CustomerID']
data = data.drop('CustomerID', axis=1)

# check if there is duplicated item
if data.duplicated().sum() != 0:
    data.drop_duplicates()
    print('All Duplicated file has been deleted')
else:
    print('There is no duplicated rows in this file')