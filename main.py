import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from fontTools.merge import cmap
from matplotlib.pyplot import legend
from scipy.spatial.transform import rotation
from scipy.stats import alpha
from seaborn import color_palette
from selenium.webdriver.support import color
from setuptools.command.rotate import rotate
# from matplotlib.pyplot import title
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
from pywaffle import Waffle





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

# <---- Calculating the correlation ---->
corr_matrix = data.select_dtypes('number').corr()
# sns.heatmap(data=corr_matrix, annot=True, fmt=".2f",linewidths=0.5)
# plt.show()

# <---- Detect outliers in the data ---->
numeric_columns = data.select_dtypes(include='number').columns
for col in numeric_columns:
    q1, q3 = data[col].quantile([0.25, 0.75])
    iqr = q3 - q1
    upper_limit = q3 + 1.5 * iqr
    lower_limit = q1 - 1.5 * iqr
    outlier = []
    for value in data[col]:
        if value > upper_limit:
            outlier.append(value)
        elif value < lower_limit:
            outlier.append(value)
    print(f'Q1 in column {col}: {q1}\nQ3 in column {col} is: {q3}\n')
    print(f'Outlier in this column:\n{outlier}\n')
    print('*'*25)

# <---- using boxplot to show outliers ---->
# plt.figure(figsize=(10,10), dpi=100)
# plt.title('This Representation to check if found  outliers')
# plt.xlabel('Features')
# plt.ylabel('count of happens')
# plt.xticks(rotation= 45, color= 'b')
# sns.boxplot(data.select_dtypes('number'))
# plt.show()

# <---- Visualize the distribution of values in each column ---->
# for col in number_column:
#     sns.displot(data=data, x=col, bins=100, facecolor='blue', kde= True, color= 'red', height=5, aspect= 3.5)
#     plt.show()

# <---- Visualizing the relation between age and payment delay ---->
# sns.scatterplot(data=data, x='Age', y='Payment Delay', hue= 'Payment Delay', palette='coolwarm')
# plt.show()

# <---- Visualizing the relation between age and total spent
# sns.scatterplot(data=data, x='Age', y='Total Spend', hue= 'Total Spend', palette='coolwarm')
# plt.show()


# <----  from this waffle I see which Feature is a controlled over other features ---->
# numeric_column_sum = data[numeric_columns].sum().tolist()
# fig_waffle = plt.figure(
#     FigureClass=Waffle,
#     rows = 5,
#     columns = 20,
#     values= numeric_column_sum,
#     legend = {'labels': numeric_columns.tolist(),
#               'loc': 'upper left',
#               'bbox_to_anchor': (1,1)},
#     figsize=(25,25)
# )
# plt.title("Waffle observation on Data")
# plt.suptitle("Total Spend feature biggest feature have unique value ")
# plt.show()

# <---- Visualizing the relation between churn and subscription type and gender, using histogram ---->
# sns.histplot(data=data,x='Churn',hue='Subscription Type', palette='Paired')
# sns.histplot(data=data,x='Churn',hue='Gender', palette='Paired')
# plt.title('Visualizing the relation between churn and subscription type and gender')
# plt.show()

# <---- Visualizing relation between numeric variables using pairplot ---->
# sns.pairplot(data.select_dtypes('number'), kind="hist", height=1.5,hue='Churn')
# plt.show()

# <---- rescaling data using MinMax Scaler ---->
numerical_features = data.select_dtypes('number')
minmax_scaler = MinMaxScaler() # create an object of scaler
scaled_numerical_features = minmax_scaler.fit_transform(numerical_features)
scaled_numerical_df = pd.DataFrame(scaled_numerical_features,columns=numeric_columns) # convert it to dataframe
# print(scaled_numerical_df.describe())

# <---- Split data to train and test ---->
encoder = LabelEncoder() # encoder object to convert categorical values to numerical values
data['Gender'] = encoder.fit_transform(data['Gender'])
scaled_numerical_df['Gender'] = data['Gender'] # add gender column to numerical features

x_train, x_test, y_train, y_test = train_test_split(scaled_numerical_df.drop('Churn', axis=1),scaled_numerical_df['Churn'], test_size=0.2, random_state=42)

# <---- Create classification models ---->
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(x_train,y_train)


# <---- Evaluate Model ---->
y_test_predict = rf_model.predict(x_test)
y_train_predict =rf_model.predict(x_train)
rf_accuracy_score = accuracy_score(y_test,y_test_predict) # Computes the percentage of correct predictions
rf_confusion_matrix = confusion_matrix(y_test,y_test_predict) # shows 4 number in order: 1-True Positive,2- False Positive, 3-False Negative, 4- True Negative
rf_report = classification_report(y_test, y_test_predict) # precision = number of correct prediction / all prediction %%%% recall = number of corr pred / all available corrects  %%% F1-socre = one value to show both of them
print(f'Accuracy of model {rf_accuracy_score}')
print('Confusion Matrix:\n', rf_confusion_matrix)
print('Classification Report:\n', rf_report)