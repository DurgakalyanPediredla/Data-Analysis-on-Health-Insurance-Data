#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 12:14:07 2024

@author: kalyanpediredla
"""

import pandas as pd
import csv as read_csv

#importing csv file into python kernal 
df= pd.read_csv("/Users/kalyanpediredla/Downloads/insurance_dataset.csv")
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns',None)

#Verifying the presence of null values in the data frame to assess its readiness for analysis.
df.notnull().all()
df.shape #checking shape of the dataframe
print(f'The insurence data contains {df.shape[0]} rows and {df.shape[1]} columns.')
print(df.isna().sum())

## Replacing missing values with most frequent value in its column.
df['medical_history']=df['medical_history'].fillna('No Medical History')
df['family_medical_history']=df['family_medical_history'].fillna('No Family Medical History')

#Displaying the initial 8 rows of the data frame for insight into the data's charactersitics.
print(df.head(8)) 
print(df.isna().sum())

##Generating statistical summaries for the numerical data columns within the data frame.
desc_stat_of_numercialdata= df.describe() 

##Generating statistical summaries for the categorical data columns within the data frame.
desc_stat_of_categoricaldata= df.describe(include=["object", "bool"]) 

print(desc_stat_of_numercialdata)
print(desc_stat_of_categoricaldata)

#Confirming distinct values in the 'region' column of the data frame.
print(df['region'].unique())  

##importing matplotlib and seaborn libraries to facilitate data visualization.
import matplotlib.pyplot as plt
import seaborn as sns

## Creating a pie chart depicting the distribution of policy holders among different regions.
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


print() ## creating space from the previous output in the console.


groupby_medical_history=  df.groupby('medical_history')['charges'].mean()

ax=groupby_medical_history.plot(kind='bar', color='skyblue',edgecolor='black')
plt.title('Average Charges by Medical History')
plt.xlabel('Medical History')
plt.ylabel('Average Charges')
for index, value in enumerate(groupby_medical_history):
    ax.text(index,value + 0.1, f'{value:.2f}', ha='center', va='bottom')
plt.show()

groupby_family_medical_history=  df.groupby('family_medical_history')['charges'].mean()

ax=groupby_family_medical_history.plot(kind='bar', color='skyblue',edgecolor='black')
plt.title('Average Charges by Family Medical History')
plt.xlabel('Family Medical History')
plt.ylabel('Average Charges')
for index, value in enumerate(groupby_family_medical_history):
    ax.text(index,value + 0.1, f'{value:.2f}', ha='center', va='bottom')
plt.show()

grouped_data= df.groupby(['medical_history','family_medical_history'])['charges'].mean().reset_index()
plt.figure(figsize=(14,10))
ax=sns.barplot(x='medical_history', y='charges',hue='family_medical_history', data=grouped_data,palette='viridis')
for p in ax.patches:
    ax.annotate(f'{p.get_height():.2f}', (p.get_x()+p.get_width()/2., p.get_height()), ha='center', va='bottom', fontsize=9)

plt.title('Average charges by medical_history and family_medical_history')
plt.xlabel('medical_history')
plt.ylabel('Average Charges')
plt.show()



# Code with the user interaction to input a threshold limit and dynaically produce results based on the provided value.
threshold_charges=int(input('Give the threshold limit : '))
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





from sklearn.preprocessing import LabelEncoder
from scipy.stats import chi2_contingency
## plotting correlation heatmap to visually show the correlation between all the categorical columns data
def chi_square_heatmap(df):
    
    categorical_cols= df.select_dtypes(include=['object']).columns
    label_encoder= LabelEncoder()
    df[categorical_cols]= df[categorical_cols].apply(label_encoder.fit_transform)
    chi2_matrix= pd.DataFrame(index=categorical_cols,columns=categorical_cols,dtype=float)
    
    for row in categorical_cols:
        for col in categorical_cols:
            if row== col:
                chi2_matrix.loc[row,col]=1.0
            else:
                contingency_table=pd.crosstab(df[row],df[col])
                _, p_value,_, _ = chi2_contingency(contingency_table)
                chi2_matrix.loc[row,col]=p_value
    plt.figure(figsize=(15,13))
    sns.set(font_scale=1.5)
    sns.heatmap(chi2_matrix, annot=True, cmap='coolwarm', fmt=".4f", linewidths=0.5,  vmin=0, vmax=1,square=True,
                cbar_kws={"shrink":0.75}, xticklabels=chi2_matrix.columns)
    plt.title("Chi-Square Test P-Values Heatmap")
    plt.show()
    
print(chi_square_heatmap(df))

## plotting correlation heatmap to visually show the correlation between all the numerical columns data.
def numerical_data_heatmap(df, numerical_columns):
    new_df=df[numerical_columns]
    correlation_matrix= new_df.corr()
    
    plt.figure(figsize=(12,10))
    sns.heatmap(correlation_matrix, annot=True, cmap='viridis', fmt='.4f', linewidths=.5)
    plt.title('Correlation Heatmap of Columns with Numerical Data')
    plt.show()

print(numerical_data_heatmap(df,['age','bmi','children','charges']))


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



















