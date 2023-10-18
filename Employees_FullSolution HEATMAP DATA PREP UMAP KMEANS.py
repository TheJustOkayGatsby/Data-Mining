import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import umap
import plotly.graph_objs as go
import plotly.figure_factory as ff

# Importing dataset and examining it
dataset = pd.read_csv("Employees.csv")
pd.set_option('display.max_columns', None)
print(dataset.head())
print(dataset.shape)
print(dataset.info())
print(dataset.describe())

# Converting categorical features to numerical features
dataset['Gender'] = dataset['Gender'].map({'Male': 1, 'Female':0})
dataset['OverTime'] = dataset['OverTime'].map({'Yes': 1, 'No':0})
dataset['BusinessTravel'] = dataset['BusinessTravel'].map({'Non-Travel':0, 'Travel_Rarely': 1, 'Travel_Frequently':2 })

# Plotting Correlation Heatmap
corrs = dataset.corr()
figure = ff.create_annotated_heatmap(
    z=corrs.values,
    x=list(corrs.columns),
    y=list(corrs.index),
    annotation_text=corrs.round(2).values,
    showscale=True)
figure.show()

# Dropping columns with high correlation + causation
dataset = dataset.drop(['YearsWithCurrManager','TotalWorkingYears','YearsSinceLastPromotion', 'PercentSalaryHike', 'JobLevel'], axis = 1)
print(dataset.info())

# Creating dummy columns
categorical_features = ['Department', 'EducationField', 'JobRole', 'MaritalStatus']
final_data = pd.get_dummies(dataset, columns = categorical_features)
print(final_data.info())
print(final_data.head(2))

# Dividing data into subsets
#Personal Data
subset1 = final_data[['Age','Gender','MaritalStatus_Single','MaritalStatus_Married','MaritalStatus_Divorced','Education','EducationField_Human Resources','EducationField_Life Sciences','EducationField_Marketing','EducationField_Medical','EducationField_Other','EducationField_Technical Degree','DistanceFromHome']]

#Work Data
subset2 = final_data[['Department_Human Resources','Department_Research & Development','Department_Sales','OverTime','StockOptionLevel','TrainingTimesLastYear','YearsAtCompany','YearsInCurrentRole','PerformanceRating']]

#Life Quality Data
subset3 = final_data[['JobSatisfaction', 'EnvironmentSatisfaction', 'JobInvolvement', 'WorkLifeBalance']]

#Potential Churn factors
subset4 = final_data[['JobSatisfaction', 'EnvironmentSatisfaction', 'JobInvolvement', 'WorkLifeBalance','OverTime','StockOptionLevel','YearsAtCompany','YearsInCurrentRole','PerformanceRating']]

# Normalizing numerical features so that each feature has mean 0 and variance 1
feature_scaler = StandardScaler()
X1 = feature_scaler.fit_transform(subset1)
X2 = feature_scaler.fit_transform(subset2)
X3 = feature_scaler.fit_transform(subset3)
X4 = feature_scaler.fit_transform(subset4)

# Analysis on subset1 - Personal Data
# Implementing UMAP to visualize dataset
u = umap.UMAP(n_neighbors=150, min_dist=0.4)
x_umap = u.fit_transform(X1)

# Finding the number of clusters (K) - Elbow Plot Method
inertia = []
for i in range(1,11):
    kmeans = KMeans(n_clusters = i, random_state = 100)
    kmeans.fit(x_umap)
    inertia.append(kmeans.inertia_)

plt.plot(range(1, 11), inertia)
plt.title('The Elbow Plot')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.show()

# Running KMeans to generate labels
kmeans = KMeans(n_clusters = 2)
kmeans.fit(x_umap)

age = list(dataset['Age'])
gender = list(dataset['Gender'])
marital = list(dataset['MaritalStatus'])
education = list(dataset['Education'])
educationfield = list(dataset['EducationField'])
distance = list(dataset['DistanceFromHome'])

data = [go.Scatter(x=x_umap[:,0], y=x_umap[:,1], mode='markers',
                    marker = dict(color=kmeans.labels_, colorscale='Rainbow', opacity=0.5),
                                text=[f'Age: {a}; Gender: {b}; MaritalStatus:{c}, Education:{d}, EducationField:{e}, DistanceFromHome:{f}' for a,b,c,d,e,f in list(zip(age,gender,marital,education,educationfield,distance))],
                                hoverinfo='text')]

layout = go.Layout(title = 'UMAP Dimensionality Reduction', width = 700, height = 700,
                    xaxis = dict(title='First Dimension'),
                    yaxis = dict(title='Second Dimension'))
fig = go.Figure(data=data, layout=layout)
fig.show()

# Analysis on subset2 - Work Data
# Implementing UMAP to visualize dataset
u = umap.UMAP(n_neighbors=150, min_dist=0.4)
x_umap = u.fit_transform(X2)

# Finding the number of clusters (K) - Elbow Plot Method
inertia = []
for i in range(1,11):
    kmeans = KMeans(n_clusters = i, random_state = 100)
    kmeans.fit(x_umap)
    inertia.append(kmeans.inertia_)

plt.plot(range(1, 11), inertia)
plt.title('The Elbow Plot')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.show()

# Running KMeans to generate labels
kmeans = KMeans(n_clusters = 2)
kmeans.fit(x_umap)

dpt = list(dataset['Department'])
ot = list(dataset['OverTime'])
so = list(dataset['StockOptionLevel'])
ttl = list(dataset['TrainingTimesLastYear'])
yac = list(dataset['YearsAtCompany'])
yic = list(dataset['YearsInCurrentRole'])
pr = list(dataset['PerformanceRating'])

data = [go.Scatter(x=x_umap[:,0], y=x_umap[:,1], mode='markers',
                    marker = dict(color=kmeans.labels_, colorscale='Rainbow', opacity=0.5),
                                text=[f'Dep: {a}; OverTime: {b}; SOL:{c}, TTL:{d}, YAC:{e}, YIC:{f}, PR:{g}' for a,b,c,d,e,f,g in list(zip(dpt,ot,so,ttl,yac,yic,pr))],
                                hoverinfo='text')]

layout = go.Layout(title = 'UMAP Dimensionality Reduction', width = 700, height = 700,
                    xaxis = dict(title='First Dimension'),
                    yaxis = dict(title='Second Dimension'))
fig = go.Figure(data=data, layout=layout)
fig.show()

# Analysis on subset3 - Life Quality Data
# Implementing UMAP to visualize dataset
u = umap.UMAP(n_neighbors=150, min_dist=0.4)
x_umap = u.fit_transform(X3)

# Finding the number of clusters (K) - Elbow Plot Method
inertia = []
for i in range(1,11):
    kmeans = KMeans(n_clusters = i, random_state = 100)
    kmeans.fit(x_umap)
    inertia.append(kmeans.inertia_)

plt.plot(range(1, 11), inertia)
plt.title('The Elbow Plot')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.show()

# Running KMeans to generate labels
kmeans = KMeans(n_clusters = 2)
kmeans.fit(x_umap)

JS = list(dataset['JobSatisfaction'])
ES = list(dataset['EnvironmentSatisfaction'])
JI = list(dataset['JobInvolvement'])
WLB = list(dataset['WorkLifeBalance'])

data = [go.Scatter(x=x_umap[:,0], y=x_umap[:,1], mode='markers',
                    marker = dict(color=kmeans.labels_, colorscale='Rainbow', opacity=0.5),
                                text=[f'JS: {a}; ES: {b}; JI:{c}, WLB:{d}' for a,b,c,d in list(zip(JS,ES,JI,WLB))],
                                hoverinfo='text')]

layout = go.Layout(title = 'UMAP Dimensionality Reduction', width = 700, height = 700,
                    xaxis = dict(title='First Dimension'),
                    yaxis = dict(title='Second Dimension'))
fig = go.Figure(data=data, layout=layout)
fig.show()

# Analysis on subset4 - Potential Churn Factors
# Implementing UMAP to visualize dataset
u = umap.UMAP(n_neighbors=150, min_dist=0.4)
x_umap = u.fit_transform(X4)

# Finding the number of clusters (K) - Elbow Plot Method
inertia = []
for i in range(1,11):
    kmeans = KMeans(n_clusters = i, random_state = 100)
    kmeans.fit(x_umap)
    inertia.append(kmeans.inertia_)

plt.plot(range(1, 11), inertia)
plt.title('The Elbow Plot')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.show()

# Running KMeans to generate labels
kmeans = KMeans(n_clusters = 2)
kmeans.fit(x_umap)

data = [go.Scatter(x=x_umap[:,0], y=x_umap[:,1], mode='markers',
                    marker = dict(color=kmeans.labels_, colorscale='Rainbow', opacity=0.5),
                                text=[f'JS: {a}; ES: {b}; JI:{c}, WLB:{d}, OverTime: {e}; SOL:{f}, YAC:{g}, YIC:{h}, PR:{i}' for a,b,c,d,e,f,g,h,i in list(zip(JS,ES,JI,WLB,ot,so,yac,yic,pr))],
                                hoverinfo='text')]

layout = go.Layout(title = 'UMAP Dimensionality Reduction', width = 700, height = 700,
                    xaxis = dict(title='First Dimension'),
                    yaxis = dict(title='Second Dimension'))
fig = go.Figure(data=data, layout=layout)
fig.show()