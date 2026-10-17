# -*- coding: utf-8 -*-
"""
Created on Sat Apr 2 13:45:13 2022

@author: Kieran Burrell
"""
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import pandas as pd

# To do linear algebra
import numpy as np

# To create plots
from matplotlib.colors import rgb2hex
from matplotlib.cm import get_cmap
import matplotlib.pyplot as plt

# To create nicer plots
import seaborn as sns

# To create interactive plots
from plotly.offline import init_notebook_mode, plot
import plotly.graph_objs as go
init_notebook_mode(connected=True)

# To get new datatypes and functions
from collections import Counter

# To investigate distributions
from scipy.stats import norm, skew, probplot

# To build models
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# To gbm light
from lightgbm import LGBMClassifier

print('Importing Datasets...')

# Get feature names
features_file = open('Dataset/features.txt', "r")
feature_list = features_file.readlines()
features_file.close()

# Load datasets
train_df = pd.read_csv('Dataset/Train/X_train.txt', sep=" ", header=None)
test_df = pd.read_csv('Dataset/Test/X_test.txt', sep=" ", header=None)

print('Datasets Imported.\n')
print('Formatting Datasets...')

# Add feature to column headers
train_df.columns = feature_list
test_df.columns = feature_list

# Get subject ID for train and test datasets
subject_id_train_file = open('Dataset/Train/subject_id_train.txt', "r")
subject_id_train_list = subject_id_train_file.readlines()
subject_id_train_file.close()
subject_id_test_file = open('Dataset/Test/subject_id_test.txt', "r")
subject_id_test_list = subject_id_test_file.readlines()
subject_id_test_file.close()

# Add subject ID
train_df['Subject'] = subject_id_train_list
test_df['Subject'] = subject_id_test_list

# Get activity ID for train and test datasets
activity_train_file = open('Dataset/Train/y_train.txt', "r")
activity_train_list = activity_train_file.readlines()
activity_train_file.close()
activity_train_list = list(map(int, activity_train_list))
activity_test_file = open('Dataset/Test/y_test.txt', "r")
activity_test_list = activity_test_file.readlines()
activity_test_file.close()
activity_test_list = list(map(int, activity_test_list))

# Get activity types list
activity_types_id = []
activity_types_name = []
activity_types_file = open('Dataset/activity_labels.txt', "r")
for line in activity_types_file:
    values = line.split()
    activity_types_id.append(int(values[0]))
    activity_types_name.append(values[1])
activity_types_file.close()

# Create activity type dictionary
activity_types_dict = {activity_types_id[i]: activity_types_name[i] for i in range(len(activity_types_id))}

# Convert activity list
activity_type_train_list = [activity_types_dict.get(item,item) for item in activity_train_list]
activity_type_test_list = [activity_types_dict.get(item,item) for item in activity_test_list]  

# Add activity
train_df['Activity'] = activity_type_train_list
test_df['Activity'] = activity_type_test_list

# Combine boths dataframe
train_df['Data'] = '0'
test_df['Data'] = '1'
both_df = pd.concat([train_df, test_df], axis=0).reset_index(drop=True)
both_df['Subject'] = '#' + both_df['Subject'].astype(str)

# Create label
label = both_df.pop('Activity')

print('Formatting Complete.\n')

print('Train Shape:\t{}'.format(train_df.shape))
print('Test Shape:\t{}\n'.format(test_df.shape))

print('-------------------')
print('Dataset Exploration')
print('-------------------\n')

print('Features Grouped by Main Names')

# Group and count main names of columns
print(pd.DataFrame.from_dict(Counter([col.split('-')[0].split('(')[0] for col in both_df.columns]), 
                             orient='index').rename(columns={0:'Count'}).sort_values('Count', ascending=False))

print('\nNull Values In DataFrame: {}\n'.format(both_df.isna().sum().sum()))
both_df.info()

print('\nCreating Plot - Smartphone Activity Label Distribution...')

# Plotting data
label_counts = label.value_counts()

# Get colours
n = label_counts.shape[0]
colourmap = get_cmap('viridis')
colours = [rgb2hex(colourmap(col)) for col in np.arange(0, 1.01, 1/(n-1))]


# Create plot
data = go.Bar(x = label_counts.index,
              y = label_counts,
              marker = dict(color = colours))

layout = go.Layout(title = 'Smartphone Activity Label Distribution',
                   xaxis = dict(title = 'Activity'),
                   yaxis = dict(title = 'Count'))

fig = go.Figure(data=[data], layout=layout)
plot(fig)

print('Plot created.\n')

print('--------------------')
print('Activity Exploration')
print('--------------------\n')

print('Creating Plots - TSNE: Activity and Participant Visualisation...')

# Create datasets
tsne_data = both_df.copy()
data_data = tsne_data.pop('Data')
subject_data = tsne_data.pop('Subject')

# Scale data
scl = StandardScaler()
tsne_data = scl.fit_transform(tsne_data)

# Reduce dimensions (speed up)
pca = PCA(n_components=0.9, random_state=3)
tsne_data = pca.fit_transform(tsne_data)

# Transform data
tsne = TSNE(random_state=3)
tsne_transformed = tsne.fit_transform(tsne_data)

# Create subplots
fig, axarr = plt.subplots(2, 1, figsize=(15,10))

### Plot Activities
# Get colors
n = label.unique().shape[0]
colormap = get_cmap('viridis')
colors = [rgb2hex(colormap(col)) for col in np.arange(0, 1.01, 1/(n-1))]

# Plot each activity
for i, group in enumerate(label_counts.index):
    # Mask to separate sets
    mask = (label==group).values
    axarr[0].scatter(x=tsne_transformed[mask][:,0], y=tsne_transformed[mask][:,1], c=colors[i], alpha=0.5, label=group)
axarr[0].set_title('TSNE: Activity Visualisation')
axarr[0].legend()

### Plot Subjects
# Get colors
n = subject_data.unique().shape[0]
colormap = get_cmap('gist_ncar')
colors = [rgb2hex(colormap(col)) for col in np.arange(0, 1.01, 1/(n-1))]

# Plot each participant
for i, group in enumerate(subject_data.unique()):
    # Mask to separate sets
    mask = (subject_data==group).values
    axarr[1].scatter(x=tsne_transformed[mask][:,0], y=tsne_transformed[mask][:,1], c=colors[i], alpha=0.5, label=group)
axarr[1].set_title('TSNE: Participant Visualisation')
plt.show()

print('Plots created.\n')
print('Testing Accuracy on Testset...\n')

# Split training testing data
enc = LabelEncoder()
label_encoded = enc.fit_transform(label)
X_train, X_test, y_train, y_test = train_test_split(tsne_data, label_encoded, random_state=3)

# Create the model
lgbm = LGBMClassifier(n_estimators=500, random_state=3)
lgbm = lgbm.fit(X_train, y_train)

# Test the model
score = accuracy_score(y_true=y_test, y_pred=lgbm.predict(X_test))
print('Accuracy on testset:\t{:.4f}\n'.format(score))

print('-----------------------')
print('Participant Exploration')
print('-----------------------\n')

print('Testing Accuracy per Activity on Testset...\n')

# Store the data
data = []
# Iterate over each activity
for activity in label_counts.index:
    # Create dataset
    act_data = both_df[label==activity].copy()
    act_data_data = act_data.pop('Data')
    act_subject_data = act_data.pop('Subject')
    
    # Scale data
    scl = StandardScaler()
    act_data = scl.fit_transform(act_data)

    # Reduce dimensions
    pca = PCA(n_components=0.9, random_state=3)
    act_data = pca.fit_transform(act_data)

    # Split training testing data
    enc = LabelEncoder()
    label_encoded = enc.fit_transform(act_subject_data)
    X_train, X_test, y_train, y_test = train_test_split(act_data, label_encoded, random_state=3)

    # Fit basic model
    print('Activity:\t\t\t\t{}'.format(activity))
    lgbm = LGBMClassifier(n_estimators=500, random_state=3)
    lgbm = lgbm.fit(X_train, y_train)
    
    score = accuracy_score(y_true=y_test, y_pred=lgbm.predict(X_test))
    print('Accuracy on testset:\t{:.4f}\n'.format(score))
    data.append([activity, score])
    
# Create duration dataframe
duration_df = (both_df.groupby([label, subject_data])['Data'].count().reset_index().groupby('Activity')
               .agg({'Data':'mean'}) * 1.28).rename(columns={'Data':'Seconds'})
activity_df = pd.DataFrame(data, columns=['Activity', 'Accuracy']).set_index('Activity')
activity_df.join(duration_df)

print('Creating Plot - Data provided by each user...')

plt.figure(figsize=(16,8))
plt.title('Data provided by each user', fontsize=20)
sns.countplot(x='Subject', data = both_df)
plt.show()

print('Plot created.\n')

print('Creating Plot - Sensor Importance For Classifying Participants By Walking Style...')

# Create dataset
tsne_data = both_df[label=='WALKING'].copy()
data_data = tsne_data
subject_data = tsne_data.pop('Subject')

# Scale data
scl = StandardScaler()
tsne_data = scl.fit_transform(tsne_data)

# Split training testing data
enc = LabelEncoder()
label_encoded = enc.fit_transform(subject_data)
X_train, X_test, y_train, y_test = train_test_split(tsne_data, label_encoded, random_state=3)

# Create model
lgbm = LGBMClassifier(n_estimators=500, random_state=3)
lgbm = lgbm.fit(X_train, y_train)

# Get importances
features = both_df.drop(['Subject'], axis=1).columns
importances = lgbm.feature_importances_

# Sum importances
data = {'Gyroscope':0, 'Accelerometer':0}
for importance, feature in zip(importances, features):
    if 'Gyro' in feature:
        data['Gyroscope'] += importance
    if 'Acc' in feature:
        data['Accelerometer'] += importance
        
# Create dataframe and plot
sensor_df = pd.DataFrame.from_dict(data, orient='index').rename(columns={0:'Importance'})
sensor_df.plot(kind='barh', figsize=(14,4), title='Sensor Importance For Classifying Participants By Walking Style (Feature Importance Sum)')
plt.show()

print('Plot created.\n')
print('Creating Plot - Participants Compared By their Staircase Walking Duration...')

# Group the data by participant and compute total duration of staircase walking
mask = label.isin(['WALKING_UPSTAIRS', 'WALKING_DOWNSTAIRS'])
duration_df = (both_df[mask].groupby([label[mask], 'Subject'])['Data'].count() * 1.28)

# Create plot
plot_data = duration_df.reset_index().sort_values('Data', ascending=False)
plot_data['Activity'] = plot_data['Activity'].map({'WALKING_UPSTAIRS':'Upstairs', 'WALKING_DOWNSTAIRS':'Downstairs'})

plt.figure(figsize=(15,5))
sns.barplot(data=plot_data, x='Subject', y='Data', hue='Activity')
plt.title('Participants Compared By Their Staircase Walking Duration')
plt.xlabel('Participants')
plt.ylabel('Total Duration [s]')
plt.show()

print('Plot created.\n')
print('Creating Plot - By What Percentage Is The Participant Faster In Walking Downstairs Than Upstairs...')

# Create data and plot
plt.figure(figsize=(15,5))
plot_data = ((duration_df.loc['WALKING_UPSTAIRS'] / duration_df.loc['WALKING_DOWNSTAIRS']) -1).sort_values(ascending=False)
sns.barplot(x=plot_data.index, y=plot_data)
plt.title('By What Percentage Is The Participant Faster In Walking Downstairs Than Upstairs?')
plt.xlabel('Participants')
plt.ylabel('Percent')
plt.show()

print('Plot created.\n')
print('Creating Plots - Staircase Walking Duration and Probability...')

def plotSkew(x):
    # Fit label to norm
    (mu, sigma) = norm.fit(x)
    alpha = skew(x)

    fig, axarr = plt.subplots(1, 2, figsize=(15,4))

    # Plot label and fit
    sns.distplot(x , fit=norm, ax=axarr[0])
    axarr[0].legend(['$\mu=$ {:.2f}, $\sigma=$ {:.2f}, $\\alpha=$ {:.2f}'.format(mu, sigma, alpha)], loc='best')
    axarr[0].set_title('Staircase Walking Duration Distribution')
    axarr[0].set_ylabel('Frequency')

    # Plot probability plot
    res = probplot(x, plot=axarr[1])
    plt.show()  
    
plotSkew(duration_df)

print('Plots created.\n')
print('Creating Plot - Participants Compared By Their Walking Duration...')

# Group the data by participant and compute total duration of walking
mask = label=='WALKING'
duration_df = (both_df[mask].groupby('Subject')['Data'].count() * 1.28)

# Create plot
plot_data = duration_df.reset_index().sort_values('Data', ascending=False)

plt.figure(figsize=(15,5))
sns.barplot(data=plot_data, x='Subject', y='Data')
plt.title('Participants Compared By Their Walking Duration')
plt.xlabel('Participants')
plt.ylabel('Total Duration [s]')
plt.show()

print('Plot created.\n')
print('Program Finished.')
