#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 18 22:02:07 2023

@author: kalyanpediredla
"""

import pandas as pd
import csv as read_csv

#importing csv file into python kernal 
df= pd.read_csv("/Users/kalyanpediredla/Downloads/insurance_dataset.csv")
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns',None)

#checking null values in dataframe.
df.notnull().all()
df.shape #checking shape of the dataframe
print(f'The insurence data contains {df.shape[0]} rows and {df.shape[1]} columns.')
print(df.isna().sum())
## replacing null values with the mode of its column.
df['medical_history'].fillna(df['medical_history'].mode()[0],inplace= True)
df['family_medical_history'].fillna(df['family_medical_history'].mode()[0],inplace= True)
print(df.head(8)) ## printing top 8 rows in the data frame.
print(df.isna().sum())
desc_stat_of_numercialdata= df.describe() ## descrptive stats of numerical data columns in the dataframe
desc_stat_of_categoricaldata= df.describe(include=["object", "bool"]) #descriptive stats of categorical data columns.

print(desc_stat_of_numercialdata)
print(desc_stat_of_categoricaldata)

print(df['region'].unique())  ##verifying unique categorical data in the 'region' column of df.

##importing matplotlib and seaborn libraries for data visualization.
import matplotlib.pyplot as plt
import seaborn as sns
## plotting pie chart that represents the distribution of policy holders across regions.
category_counts=df['region'].value_counts()
explode=(0,0,0,0)
colors=['gold','lightcoral','lightskyblue','lightgreen']
plt.subplots()
plt.pie(category_counts, labels=category_counts.index,autopct='%1.1f%%',startangle=90,pctdistance=0.85, explode=explode,colors=colors)
centre_circle=plt.Circle((0,0),0.70,fc='white')
fig=plt.gcf()
fig.gca().add_artist(centre_circle)
plt.axis('equal')
plt.legend(category_counts.index,title='Categories',loc='upper right')
plt.title('Distribution of policy holders across regions')
plt.show()

## importing label encoder library to assign numerical values to unique categorical data.
## plotting correlation heatmap to visually show the correlation between all the columns data.
from sklearn.preprocessing import LabelEncoder
def correlation_heatmap():
    x=df.select_dtypes(include=['object']).columns
    y=LabelEncoder()
    for column in x:
        df[column]=y.fit_transform(df[column])
    correlation_matrix=df.corr()
    plt.figure(figsize=(10,8))
    sns.set(font_scale=1.2)
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f",linewidths=0.5,vmin=-1,vmax=1,square= True, cbar_kws={"shrink":0.75},
                xticklabels=correlation_matrix.columns, yticklabels=correlation_matrix.columns,)
    plt.title("Correlation Heatmap")
    plt.show()
correlation_heatmap()


## asking the user to give a threshold limit to proceed with the data visualization.
print() ## creating space from the previous output in the console.
threshold_charges=int(input('Give the threshold limit : '))
df= pd.read_csv("/Users/kalyanpediredla/Downloads/insurance_dataset.csv")

##plotting violin plot that represents the probability of charges exceeding the threshold level.(eg.$15000)
def prob_charges_exceeding_thershold_by_region(threshold_charges):
    selected_regions=['northwest', 'southeast', 'southwest']
    selected_df = df[df['region'].isin(selected_regions)]
    total_records=len(selected_df)
    exceed_thereshold_records= len(selected_df[selected_df['charges']>threshold_charges])
    probability_exceeding_threshold= exceed_thereshold_records/total_records
    print(f"Probability of charges exceeding $15000 for selected regions: {probability_exceeding_threshold: .2%}")
    plt.figure(figsize=(12,8))
    colors=sns.color_palette('viridis', n_colors=len(selected_regions))
    ax=sns.violinplot(x='region' ,y='charges', data= selected_df, inner='quartile',palette=colors)
    sns.set_theme(style='whitegrid')
    ax.axhline(y=threshold_charges , color='red', linestyle='--', label= f'Threshold ({threshold_charges})')
    plt.xlabel('Region', fontsize=14)
    plt.ylabel('Charges', fontsize=14)
    plt.title(f'Probability of Charges Exceeding a Threshold of {threshold_charges} by Region is {probability_exceeding_threshold: .2%}',fontsize=16)
    ax.legend()
    plt.show()
prob_charges_exceeding_thershold_by_region(threshold_charges)

##plotting the barplot that represents the average BMI by region.
def plot_avg_bmi_by_region(df,bmi,region):
    average_bmi_by_region= df.groupby(region)[bmi].mean().reset_index()
    print(average_bmi_by_region)
    plt.figure(figsize=(10,6))
    colors=sns.color_palette("BuPu",n_colors=len(average_bmi_by_region))
    ax=sns.barplot(x=region,y=bmi, data=average_bmi_by_region,palette=colors)
    ax.set(xlabel='Region', ylabel='Average BMI', title='Average BMI by Region')
    for index, value in enumerate((average_bmi_by_region[bmi])):
        ax.text(index, value + .01, f'{value:.2f}',ha='center', va='bottom', fontsize=10)
    plt.show()   
plot_avg_bmi_by_region(df, 'bmi', 'region')


##plotting the barplot that represents the average charges by coverage level.
def plot_avg_charges_by_coverage(df,charges,coverage_level):
    average_charges_by_coverage= df.groupby(coverage_level)[charges].mean().reset_index()
    print(average_charges_by_coverage)    
    plt.figure(figsize=(10,6))
    colors=sns.color_palette("husl",n_colors=len(average_charges_by_coverage))
    ax=sns.barplot(x=coverage_level,y=charges, data=average_charges_by_coverage,palette=colors,)
    sns.set_theme(style='whitegrid')
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)
    ax.set(xlabel='Coverage Level', ylabel='Average Charges', title='Average Charges for each Coverage Type')
    ax.set_xticklabels(average_charges_by_coverage[coverage_level], ha='right', fontsize=12)
    for index, value in enumerate((average_charges_by_coverage[charges])):
        ax.text(index, value + 50, f'{value:.2f}',ha='center', va='bottom', fontsize=10)      
    plt.show()
plot_avg_charges_by_coverage(df,'charges', 'coverage_level')


##plotting the barplot that represents the average age by smoking status.
def plot_avg_age_by_smoking_status(df,age,smoker):
    average_age_by_smoking_status= df.groupby(smoker)[age].mean().reset_index()
    print(average_age_by_smoking_status)
    
    plt.figure(figsize=(10,6))
    
    colors=sns.color_palette('Greys',n_colors=len(average_age_by_smoking_status))
    ax=sns.barplot(x=smoker,y=age, data=average_age_by_smoking_status,palette=colors)
    sns.set_theme(style='whitegrid')
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)
    ax.set(xlabel='Smoker?', ylabel='Average Age', title='Average Age of Smoker and Non-Smoker')
    for index, value in enumerate((average_age_by_smoking_status[age])):
        ax.text(index, value + .01, f'{value:.2f}',ha='center', va='bottom', fontsize=10)      
    plt.show()
plot_avg_age_by_smoking_status(df, 'age', 'smoker')

## plotting scatter plot with regression line that shows the relationship between BMI and Charges.
def bmi_vs_charges_plot(df, bmi, charges):
    plt.figure(figsize=(10,6))
    sns.regplot(x=bmi,y=charges,data=df,scatter_kws={'s':10,'alpha':0.5},line_kws={'color':'red'})
    sns.set_theme(style='whitegrid')
    plt.xlabel('BMI', fontsize=14)
    plt.ylabel('Charges', fontsize=14)
    plt.title('Scatter Plot:BMI vs Charges',fontsize=16)
    plt.show()
bmi_vs_charges_plot(df, 'bmi', 'charges')


### plotting barplot that represents average charges by gender.
def compare_charges_by_gender(df, charges, gender):
    average_charges_by_gender=df.groupby(gender)[charges].mean().reset_index()
    print(average_charges_by_gender)
    
    plt.figure(figsize=(10,6))
    colors=sns.color_palette("Set2",n_colors=len(average_charges_by_gender))
    sns.barplot(x=gender,y=charges, data=average_charges_by_gender,palette=colors)
    sns.set_theme(style='whitegrid')
    plt.xlabel('Gender', fontsize=14)
    plt.title('Average Charges by Gender', fontsize=16)
    for index, value in enumerate (average_charges_by_gender[charges]):
        plt.text(index, value + .01 , f'{value:.2f}', ha= 'center', va='bottom', fontsize=12)
    plt.show()
compare_charges_by_gender(df, 'charges', 'gender')


### plotting boxplot that represents the distribution of charges by coverage level.
def distribution_of_charges_by_coverage_level(df, coverage_level,charges):
    plt.figure(figsize=(10,6))
    colors= sns.color_palette("Set3",n_colors=len(df[coverage_level].unique()))
    sns.boxplot(x=coverage_level,y=charges, data=df, palette=colors)
    sns.set_theme(style='whitegrid')
    plt.xlabel('Coverage Level', fontsize=14)
    plt.ylabel('Charges',fontsize=16)
    plt.title('Distribution of Charges by Coverage Level', fontsize=16)
    plt.show()

distribution_of_charges_by_coverage_level(df, 'coverage_level', 'charges')


# In[ ]:




