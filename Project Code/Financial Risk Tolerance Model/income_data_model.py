#!/usr/bin/env python
# coding: utf-8

# In[31]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
import pickle


# In[2]:


datadir_final = 'F:/Masters/MRP/Datasets/'


# In[4]:


income = pd.read_csv(datadir_final+'Income_Data/Income_data.csv')
income


# In[5]:


income.columns


# In the below cell I am replacing all the nan values present in the occupation_type column with random occupations in order to use occupation_type as a parameter for calculating the Financial Risk Tolerance index of an individual.

# In[3]:


non_nan_values = income['Occupation_type'].dropna().values
def fill_nan_with_random(column):
    nan_indices = column[column.isna()].index
    random_values = np.random.choice(non_nan_values, size=len(nan_indices))
    column.loc[nan_indices] = random_values
    return column
income['Occupation_type'] = fill_nan_with_random(income['Occupation_type'])


# In[7]:


income['Cost_of_living'] = income['Annual_income'] / (income['Family_Members']+1) #Adding a new column Cost of living which stores the cost of living an individual can afford for it's family member annually.
income.head()


# In[8]:


income['Occupation_type'].unique()


# Due to different type of occupation types, I am catogorizing all the different Occupation types into 3 categories, namely Professional/Management, Skilled Labour and Unskilled Labour and storing them into a new column named Occupation_category.
# 
# Below shows details of each cateorgy of occupation:
# 
# 1. Professional/Management: Managers, High skill tech staff, Accountants, Medicine Staff, HR Staff, IT Staff	
# 
# 2. Skilled Labour: Sales staff, Core staff, Security staff, Cooking staff, Private service staff, Secreatries, Realty agents	
# 
# 3. Unskilled Labour: Laborers, Drivers, Cleaning staff, Low-skill labores, Waiters/barmen staff

# In[15]:


occupation_category = []
professional = ['Managers', 'High skill tech staff', 'Accountants', 'Medicine staff', 'HR staff', 'IT staff']
skilled = ['Sales staff', 'Core staff', 'Security staff', 'Cooking staff', 'Private service staff', 'Secretaries', 'Realty agents']
unskilled = ['Laborers', 'Drivers', 'Cleaning staff', 'Low-skill Laborers', 'Waiters/barmen staff']
for i in income['Occupation_type']:
    if i in professional:
        occupation_category.append('Professional/Management')
    elif i in skilled:
        occupation_category.append('Skilled Labour')
    elif i in unskilled:
        occupation_category.append('Unskilled Labour')

income['Occupation_category'] = occupation_category
income


# The way we categorized the Occupation_type, in the same way we will categorize the Education and divide them into three education levels, Higher Education, Secondary Education, Lower Secondary Education and storing it into a new column 'Education_category'
# 
# The category we are dividing is as follows:
# 1. Higher Education: Higher education, Academic degree
# 2. Secondary Education: Secondary / secondary special, Incomplete higher
# 3. Lower Secondary Education: Lower secondary

# In[21]:


education_category = []
higher = ['Higher education', 'Academic degree']
secondary = ['Secondary / secondary special', 'Incomplete higher']
lower = ['Lower secondary']
for i in income['Education']:
    if i in higher:
        education_category.append('Higher Education')
    elif i in secondary:
        education_category.append('Secondary Education')
    elif i in lower:
        education_category.append('Lower Secondary Education')

income['Education_category'] = education_category
income


# In[23]:


income.to_csv(datadir_final+'Income_Data/Income_data_updated.csv') #Saving the csv file for future use


# I created a new column Financial Risk Tolerance Score which actually scores an individual financial risk tolerance capacity. The higher the score, the higher the ability of an individual to take risk financially. This score has been created by me using fusion of multiple factors and assigning different weights to each parameter used to influence this score. The maximum score an individual can acheive is 1500. The way the score has been calculated is mentioned in this excel file: https://docs.google.com/spreadsheets/d/1qSEq9vjAceuxV63vu1yx5EMO3iOYBFLw/edit?usp=sharing&ouid=102980949826867451781&rtpof=true&sd=true

# In[76]:


def risk_score_costofliving(cost_of_living):
    weight = 3
    if cost_of_living < 50000:
        return weight*10
    elif cost_of_living >= 50000 and cost_of_living < 100000:
        return weight*25
    elif cost_of_living >= 100000 and cost_of_living < 150000:
        return weight*40
    elif cost_of_living >= 150000 and cost_of_living < 200000:
        return weight*60
    elif cost_of_living >= 200000 and cost_of_living < 250000:
        return weight*80
    elif cost_of_living >= 250000:
        return weight*100

def risk_score_carowner(car_owner):
    weight = 1
    if car_owner == 'Y':
        return weight*100
    else:
        return weight*50

def risk_score_property(property_owner):
    weight = 3
    if property_owner == 'Y':
        return weight*100
    else:
        return weight*25

def risk_score_employed(employed):
    weight = 2
    if employed == 'Y':
        return weight*100
    else:
        return weight*25

def risk_score_maritialstatus(status):
    weight = 1
    if status == 'Married' or status == 'Civil marriage':
        return weight*50
    elif status == 'Single / not married':
        return weight*100
    elif status == 'Separated':
        return weight*25
    elif status == 'Widow':
        return weight*10

def risk_score_age(age):
    weight = 1
    if age < 25:
        return weight*100
    elif age >= 25 and age < 40:
        return weight*80
    elif age >= 40 and age < 50:
        return weight*60
    elif age >= 50 and age < 60:
        return weight*40
    elif age >= 60:
        return weight*20
    
def risk_score_housetype(house_type):
    weight = 2
    if house_type == 'House / apartment':
        return weight*100
    elif house_type == 'With parents' or house_type == 'Co-op apartment':
        return weight*90
    elif house_type == 'Municipal apartment':
        return weight*25
    elif house_type == 'Rented apartment':
        return weight*40
    elif house_type == 'Office apartment':
        return weight*60

def risk_score_occupation_education(occupation, education):
    weight = 2
    if occupation == 'Professional/Management':
        if education == 'Higher Education':
            return weight*100
        elif education == 'Secondary Education':
            return weight*75
        elif education == 'Lower Secondary Education':
            return weight*50
    elif occupation == 'Skilled Labour':
        if education == 'Higher Education':
            return weight*90
        elif education == 'Secondary Education':
            return weight*60
        elif education == 'Lower Secondary Education':
            return weight*30
    elif occupation == 'Unskilled Labour':
        if education == 'Higher Education':
            return weight*50
        elif education == 'Secondary Education':
            return weight*30
        elif education == 'Lower Secondary Education':
            return weight*15


# In[83]:


financial_risk = []
for index, row in income.iterrows():
    risk_score = 0
    risk_score += risk_score_costofliving(row['Cost_of_living'])
    risk_score += risk_score_carowner(row['Car_Owner'])
    risk_score += risk_score_property(row['Property_Owner'])
    risk_score += risk_score_employed(row['Employed'])
    risk_score += risk_score_maritialstatus(row['Marital_status'])
    risk_score += risk_score_age(row['Age'])
    risk_score += risk_score_housetype(row['Housing_type'])
    risk_score += risk_score_occupation_education(row['Occupation_category'], row['Education_category'])
    financial_risk.append(risk_score)

income['Financial_risk_tolerance_score'] = financial_risk


# After calculating the Financial Risk Tolerance Score, I divided all the individuals into different risk categories, based on their financial risk tolerance score calculated in the cell above. 
# 
# I have created six different risk categories as follows:
# 1. Very High Risk: Score >= 1250
# 2. High Risk: 1250 < Score >= 1100
# 3. Medium to High Risk: 1100 < Score >= 950
# 4. Medium Risk: 950 < Score >= 850
# 5. Low to Medium Risk: 850 < Score >= 700
# 6. Low Risk: Score < 700

# In[4]:


def calculate_risk_category(score):
    if score >= 1250:
        return 'Very High Risk'
    elif score < 1250 and score >= 1100:
        return 'High Risk'
    elif score < 1100 and score >= 950:
        return 'Medium to High Risk'
    elif score < 950 and score >= 850:
        return 'Medium Risk'
    elif score < 850 and score >= 700:
        return 'Low to Medium Risk'
    elif score < 700:
        return 'Low Risk'


# In[5]:


risk_level = []
for i in income['Financial_risk_tolerance_score']:
    risk_level.append(calculate_risk_category(i))

income['Risk_category'] = risk_level


# In[6]:


income.to_csv(datadir_final+'Income_Data/Income_data_updated_2.csv') #Saving the csv file for future use


# In[8]:


updated_income = income[['Gender', 'Car_Owner', 'Property_Owner', 'Marital_status', 'Housing_type', 'Age', 'Employed', 'Cost_of_living', 'Occupation_category', 'Education_category', 'Financial_risk_tolerance_score']]
#creating a new dataframe with the required columns for DecisionTree Regression


# In[9]:


updated_income


# In[10]:


def mapper_to_categorical(data_row):
    gender = ['M', 'F']
    car = ['Y', 'N']
    prop = ['Y', 'N']
    maritial = ['Married', 'Single / not married', 'Civil marriage', 'Separated', 'Widow']
    house = ['House / apartment', 'With parents', 'Rented apartment', 'Municipal apartment', 'Co-op apartment', 'Office apartment']
    occupation = ['Unskilled Labour', 'Professional/Management', 'Skilled Labour']
    education = ['Higher Education', 'Secondary Education', 'Lower Secondary Education']
    return [gender.index(data_row[0]), car.index(data_row[1]), prop.index(data_row[2]), 
            maritial.index(data_row[3]), house.index(data_row[4]), data_row[5], data_row[6],
            data_row[7], occupation.index(data_row[8]), education.index(data_row[9]),
            data_row[10]]

# Mapper function above helps to convert text columns to catogorical for Regressor model.


# In[11]:


dataset = []
for row in updated_income.values.tolist():
    dataset.append(mapper_to_categorical(row))
new_income = pd.DataFrame(dataset, columns=updated_income.columns)
new_income #new income dataframe with catogorical values, will be used as an input for the regressor model


# ### Decision Tree Regressor

# In[32]:


X = new_income[['Gender', 'Car_Owner', 'Property_Owner', 'Marital_status', 'Housing_type', 'Age', 'Employed', 'Cost_of_living', 'Occupation_category', 'Education_category']]
y = new_income['Financial_risk_tolerance_score']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=25)

model = DecisionTreeRegressor(random_state=25)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_pred_rounded = np.round(y_pred).astype(int)

rmse = np.sqrt(mean_squared_error(y_test, y_pred_rounded))
r2 = r2_score(y_test, y_pred)
print(f'Root Mean Squared Error: {rmse}')
print(f'R2 Score: {r2}')

y_test_risk_category = [calculate_risk_category(i) for i in y_test]
y_pred_risk_category = [calculate_risk_category(i) for i in y_pred_rounded]

accuracy = accuracy_score(y_test_risk_category, y_pred_risk_category)
print(f'Accuracy: {accuracy} \n')

file_path = 'decision_tree_income.pkl'
with open(file_path, 'wb') as file:
    pickle.dump(model, file) #Saving model for future use

comparison = pd.DataFrame({'Actual': y_test, 'Actual_risk_category': y_test_risk_category, 'Predicted': y_pred_rounded, 'Predicted_risk_category': y_pred_risk_category})
comparison


# ### Random Forest Regressor

# In[33]:


X = new_income[['Gender', 'Car_Owner', 'Property_Owner', 'Marital_status', 'Housing_type', 'Age', 'Employed', 'Cost_of_living', 'Occupation_category', 'Education_category']]
y = new_income['Financial_risk_tolerance_score']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=25)

model = RandomForestRegressor(n_estimators=250, random_state=25)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_pred_rounded = np.round(y_pred).astype(int)

rmse = np.sqrt(mean_squared_error(y_test, y_pred_rounded))
r2 = r2_score(y_test, y_pred)
print(f'Root Mean Squared Error: {rmse}')
print(f'R2 Score: {r2}')

y_test_risk_category = [calculate_risk_category(i) for i in y_test]
y_pred_risk_category = [calculate_risk_category(i) for i in y_pred_rounded]

accuracy = accuracy_score(y_test_risk_category, y_pred_risk_category)
print(f'Accuracy: {accuracy} \n')

file_path = 'random_forest_income.pkl'
with open(file_path, 'wb') as file:
    pickle.dump(model, file) #Saving model for future use

comparison = pd.DataFrame({'Actual': y_test, 'Actual_risk_category': y_test_risk_category, 'Predicted': y_pred_rounded, 'Predicted_risk_category': y_pred_risk_category})
comparison


# ### Gradient Boosting Machine(GBM) Regressor

# In[34]:


X = new_income[['Gender', 'Car_Owner', 'Property_Owner', 'Marital_status', 'Housing_type', 'Age', 'Employed', 'Cost_of_living', 'Occupation_category', 'Education_category']]
y = new_income['Financial_risk_tolerance_score']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=25)

model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=25)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_pred_rounded = np.round(y_pred).astype(int)

rmse = np.sqrt(mean_squared_error(y_test, y_pred_rounded))
r2 = r2_score(y_test, y_pred)
print(f'Root Mean Squared Error: {rmse}')
print(f'R2 Score: {r2}')

y_test_risk_category = [calculate_risk_category(i) for i in y_test]
y_pred_risk_category = [calculate_risk_category(i) for i in y_pred_rounded]

accuracy = accuracy_score(y_test_risk_category, y_pred_risk_category)
print(f'Accuracy: {accuracy} \n')

file_path = 'gbm_income.pkl'
with open(file_path, 'wb') as file:
    pickle.dump(model, file) #Saving model for future use
    
comparison = pd.DataFrame({'Actual': y_test, 'Actual_risk_category': y_test_risk_category, 'Predicted': y_pred_rounded, 'Predicted_risk_category': y_pred_risk_category})
comparison


# From all the above three models Decision Tree, Random Forest and Gradient Boosting Machine (GBM), GBM gave the highest r2_score and accuracy in predicting risk_category. I will be proceeding with GBM for predicting individuals Financial Tolerance level.
