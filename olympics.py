
##########     importing libraries    #########

import numpy as np
import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split 
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

#setting options for better visualization
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_colwidth', 100)

############## Title  ##############

st.header('120 years of Olympic history')
st.image('https://images.news18.com/ibnlive/uploads/2021/06/1625046172_sports-14.png?im=Resize,width=360,aspect=fit,type=normal?im=Resize,width=320,aspect=fit,type=normal')
st.write("Programming Project - University of Verona")


st.subheader('About Dataset') 

st.write("Context:")
st.write("This is a historical dataset on the modern Olympic Games, including all the Games from Athens 1896 to Rio 2016. data has scraped from www.sports-reference.com in May 2018.The Winter and Summer Games were held in the same year up until 1992. After that, they staggered them such that Winter Games occur on a four year cycle starting with 1994, then Summer in 1996, then Winter in 1998, and so on.")
st.write("Content:")
st.write("The file athlete_events.csv contains 271116 rows and 15 columns. Each row corresponds to an individual athlete competing in an individual Olympic event (athlete-events).") 
st.write("The columns are:")
st.write("* ID : Unique number for each athlete \n* Name : Athlete name \n* Sex : M or F \n* Age : Integer \n* Height : In centimeters \n* Weight : In kilograms \n* Team : Team name \n* NOC : National Olympic Committee 3-letter code \n* Games : Year and season \n* Year : Integer \n* Season : Summer or Winter \n* City : Host city \n* Sport : Sport \n* Event : Event \n* Medal : Gold, Silver, Bronze, or NA")

############## Loading the dataset ##############

#loading the dataset
df_noc = pd.read_csv('https://raw.githubusercontent.com/MaliheDahmardeh/Olympic-History/main/noc_regions.csv')
df_events = pd.read_csv("C:/Users/habib/Desktop/athlete_events.csv")
df = pd.merge(df_events,df_noc,on='NOC',how='left')

st.subheader('Dataset') 
if st.checkbox('show the raw data'):
   st.write(df)

####################          Data cleaning          ##################

df.drop(columns=['notes','ID'],inplace=True)
#replace missing data in the regions columns
df['region'] = np.where(df['NOC']=='SGP', 'Singapore', df['region'])
df['region'] = np.where(df['NOC']=='TUV', 'Tuvalu', df['region'])
df['region'] = np.where(df['NOC']=='UNK', 'Unknown', df['region'])
df['region'] = np.where(df['NOC']=='ROT', 'Refugee Olympic Athletes', df['region'])
#drop duplicates
df.drop_duplicates(inplace=True)
#Mean of height
mean_height = df["Height"].mean()
rmh = round(mean_height)
df["Height"] = df["Height"].fillna(rmh)
#Mean of weight
mean_weight = df["Weight"].mean()
rmw = round(mean_weight)
df["Weight"] = df["Weight"].fillna(rmw)
#Mean of Age
mean_age = df["Age"].mean()
rma = round(mean_age)
df["Age"] = df["Age"].fillna(rma)
#Changing float type data to integer
df["Age"] = df["Age"].astype(int)
df["Height"] = df["Height"].astype(int)
df["Weight"] = df["Weight"].astype(int)

#making a copy of dataframe with medal NaN value for prediction part
new_df = df.copy()

#drop Medal null values
df.dropna( how ='any',subset = ['Medal'], inplace = True)


###################            Data Exploration and Visualization           ####################

st.title('Overall Statistic')

years = df['Year'].nunique()
cities = df['City'].nunique()
sports = df['Sport'].nunique()
events = df['Event'].nunique()
athletes = df['Name'].nunique()
countries = df['region'].nunique()

col1, col2, col3  = st.columns(3)
with col1:
     st.header('Years')
     st.title(years)
with col2:
     st.header('Hosts')
     st.title(cities)
with col3:
     st.header('Sports')
     st.title(sports)

col1, col2, col3  = st.columns(3)
with col1:
     st.header('Events')
     st.title(events)
with col2:
     st.header('Nations')
     st.title(countries)
with col3:
     st.header('Athletes')
     st.title(athletes)

col1, col2  = st.columns(2)
with col1:
  #total number of Gold,Silver,Bronze Medals during the 1896-2016
     medals = df['Medal'].value_counts()
     st.write('Total number of different medals:')
     st.write(medals)
with col2:
  #compare number of "Team", "NOC", "Event", "City", "Sport" for different season
     season_count = df.groupby("Season")[["Team", "NOC", "Event", "City", "Sport"]].nunique().reset_index()
     st.write('Season Comparison')
     st.write(season_count)


#Medals is a categorical column we can get separate columns of Gold, Silver and Bronze from column Medal by using the get_dummies method of pandas.
df = pd.concat([df,pd.get_dummies(df['Medal'])],axis=1)


#################    participation of nations in Olympic over the year ###################

st.subheader('Participation of nations in Olympics over the years') 

nations1_df =df[df['Season']== 'Summer']
nations1 = nations1_df.groupby('Year').count()['region'].reset_index()
nations1.rename(columns={"region":"Count"},inplace=True)
fig= px.line(nations1,x ='Year',y ='Count')
fig.update_layout(title='<b>Summer Olympics<b>',plot_bgcolor = " whitesmoke")
st.write(fig)


nations2_df =df[df['Season']== 'Winter']
nations2= nations2_df.groupby('Year').count()['region'].reset_index()
nations2.rename(columns={"region":"Count"},inplace=True)
fig= px.line(nations2,x ='Year',y ='Count')
fig.update_layout(title='<b>Winter Olympics<b>',plot_bgcolor = " whitesmoke")
st.write(fig)


#############      PERFORMANCE OF COUNTRIES IN THE OLYMPICS       #########

st.subheader('Performance of countries in olympics')

df_medal_duplicated = df.drop_duplicates(subset=['Event','Sport','Team','region','Games','Year','City','Medal']).reset_index()
df_medal_duplicated.drop(columns=['index'], inplace=True)
#team sport medals change to one for each country
df_medal_duplicated = df.drop_duplicates(subset=['Event','Sport','Team','region','Games','Year','City','Medal']).reset_index()
df_medal_duplicated.drop(columns=['index'], inplace=True)
#number of different medals from 1896 to 2016 for each region,by using (df_medal_duplicated) for counting medal for different country consider one medal for each team sport,
number_of_different_medals=df_medal_duplicated.groupby('region').sum()[['Gold','Silver','Bronze']].reset_index()
number_of_different_medals['Total']=number_of_different_medals['Gold'] + number_of_different_medals['Silver'] + number_of_different_medals['Bronze']
country_medals=number_of_different_medals.sort_values('Total',ascending=False).reset_index()
country_medals.drop(columns=['index'], inplace=True)
medals_region=country_medals.head(20)

#by bar chart we see the 20 team that have the highest number of medals in Olympic
fig = px.bar(medals_region, x='region', y='Total', 
      labels={"region": "Name of the Countries",
              "Total": "Total number for medals in each Country"
              })
fig.update_layout(title='<b>Number of medals for each country<b>',height=700)
st.write(fig)


#############    Sports and Medals      ############

#we see the number of Medals/athletes in each sport that helds in Olympics
df_sport = df.groupby('Sport')['Name'].count().reset_index(name = 'Count')
df_sport = df_sport.sort_values('Count',ascending = False).reset_index()
s = df_sport.head(29)
#Medals won in each sport in the Olympics
fig = px.bar(s, x='Sport', y='Count', 
      labels={"Sport": "Name of the Sports",
              "Count": "Total number of medals in each sport"
              })
fig.update_layout(title='<b>Number of medals for each sport<b>',height=700)
st.write(fig)

################  Top Athletes and Countries  ################

st.subheader('Top-20 countries were successful in what sports?')

st.write("* USA, Russia, Germany and UK have their highest number of medals in Athletics sport. \n* USA stand at the first place with 816 medals and after that Russia and Germany have taken the second and thirs place respectively with 491 and 264 medals. \n* Most popular sports are Ayhletics and Swimming. ")
 #making a matrix of region and sport 
country_sport=df_medal_duplicated .groupby('region')['Sport'].value_counts().sort_values(ascending=False).unstack().fillna(0)
 #top 20 countries with most medals in one sport
x=country_sport.max(axis=1).sort_values(ascending=False).head(20)
 #finding for each country the name of successfull sport
y=country_sport.idxmax(axis=1)
 #define series for x and y
series1 = pd.Series(x , name='Medals')
series2 = pd.Series(y , name='Sport')
 #merge series into DataFrame
df_xy = pd.concat([series1, series2], axis=1)
xy=df_xy.head(20)

if st.checkbox('Show details:'):
   st.write(xy)




###########   Performane of Men and women   ###########


st.subheader('Performance of Men and Women')
col1, col2  = st.columns(2)
with col1:
  #  we see the percentage of men and women who won the medal in Olympic    ########## pie plot  ########
  fig = plt.figure(figsize =(3,3))
  df_pie=df['Sex'].value_counts()
  df_pie.plot.pie(explode=[0,0.1],autopct='%.2f', shadow =True , colors=('lightblue','pink'))
  st.write(fig)
  st.caption('Percentage of Men and Women who won medals')
with col2:
  #number of women who have won medals at the summer Olympics
  women = df[(df['Sex'] == 'F')& (df['Season'] == 'Summer')].groupby('Year').count()['Name'].reset_index()
  #number of men who have won medals at the summer Olympics
  men = df[(df['Sex'] == 'M')& (df['Season'] == 'Summer')].groupby('Year').count()['Name'].reset_index()
  #number of men and women with medals in each year
  total_athlete= men.merge(women,on='Year',how='left')
  total_athlete.rename(columns={'Name_x':'Male','Name_y':'Female'},inplace=True)
  total_athlete = total_athlete.fillna(0)

  #men and women performance over the time for SUMMER olympic season
  fig = px.line(total_athlete,x='Year',y=['Male','Female'])
  fig.update_layout(title='<b>Performance of men and women in summer olympics over the years<b> ',plot_bgcolor = 'whitesmoke')
  st.write(fig)

#  number of different medals for men and women   ####    Line chart    #####
men_and_women_medals=df.groupby('Sex').sum()[['Gold','Silver','Bronze']].sort_values('Gold',ascending=False).reset_index()

if st.checkbox('men_and_women_medals'):
    st.write(men_and_women_medals)



###########   TOP Contries and Athletes  ###########
st.subheader('TOP Contries and Athletes ')

 #define a function to find top athletes for each country
def top_country_athletes(df,country):
     tmw_df = df.dropna(subset=['Medal'])
     tmw_df = tmw_df[tmw_df['region'] == country]
     x = tmw_df['Name'].value_counts().reset_index().head(15)
     x = x.merge(df,left_on='index',right_on='Name',how='left')[['index','Name_x','Sport']].drop_duplicates()
     x.rename(columns={'index':'Name','Name_x':'Medals'},inplace=True)
     return x
 #using function for USA as first country in the medal winners    
y = top_country_athletes(df,'USA')
y = y.reset_index().head(15)
y.drop(columns=['index'], inplace=True)

  #Top-15 athlete of USA
fig = px.bar(y, x='Name', y='Medals', color='Sport', labels={
            "Name": "Name of the athletes",
            "Medals": "Total medals won by the athletes",
            "Sport": "Sports played by the athletes"
            },)
fig.update_layout(title='<b>Top-15 Athlete of USA<b>', height=700)
st.write(fig)



#checking for top 15 medal winner in olympics
t_10 = df['Name'].value_counts().reset_index()
t_10  = t_10 .merge(df, left_on='index', right_on='Name')[['index', 'Name_x', 'Sport', 'region']].drop_duplicates()
t_10 .rename(columns={'index': 'Name', 'Name_x': 'Medals'}, inplace=True)
t_10  = t_10 .reset_index().head(15)
t_10 .drop(columns=['index'], inplace=True)

  #Top-10 Successfull athletes
fig = px.bar(t_10, x='Name', y='Medals', color='region', labels={
        "Name": "Name of the athlete",
        "Medals": "Total medals won by the athlete",
        "region": "Country of the athlete"
          })
fig.update_layout(title='<b>Top-15 medal winners in the olympics<b>',height=700)
st.write(fig)




#############          Height,Weight,Age        ############
st.subheader('Height ,Weight and Age Variation')
#scatterplot for distribution of height weight age
fig = px.scatter(df, x="Height", y="Weight",color='Age')
fig.update_layout(autosize=True)
st.write(fig)



################       Prediction       ###############

st.subheader('Prediction')
 ###encoding
data=new_df.copy()
data.drop(['Name','Games','region'],axis=1,inplace=True)

 #creating list of numerical columns
numerical_columns = [column for column in data.columns if ((data.dtypes[column] != 'object') & (column not in ['Medal','Year']))]
 #creating list of categorical columns
categorical_columns = [column for column in data.columns if data.dtypes[column] == 'object']

 #for medals consider all the gold,silver and bronze medals 1 and other null value 0
data['Medal'] = data['Medal'].apply(lambda x: 1 if str(x) != 'nan' else 0)

 # Binary encoding function for categorical columns
def binary_encoder(data, columns, positive_values):
      df = data.copy()

      for column, positive_value in zip(columns, positive_values):
        df[column] = df[column].apply(lambda x: 1 if x == positive_value else 0)
      return df

  # using Binary encoding for columns with 2 categories
data = binary_encoder(data, columns=['Sex', 'Season'], positive_values=['M', 'Summer'])

  # one hot encoder function for categorical columns
def onehot_encoder(data, columns):
      df = data.copy()
      for column in columns:
        dummies = pd.get_dummies(df[column],drop_first=True)
        # concat, dummies , original dataframe
        df = pd.concat([df, dummies], axis=1)
        
        # dropping original columns for which encoding applied.
        df.drop(column, axis=1,inplace=True)
      return df

   # applying onehot encoding on features with more than 2 categories
data = onehot_encoder(data,columns = ['Team', 'NOC', 'City', 'Sport', 'Event'])

 # Split dataframe into variables, dependent(y) and indepedent(X)
X = data.drop('Medal',axis=1).copy()
y = data['Medal'].copy()

 # Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=50)

 # Scaling Numerical Features
s_scaler = StandardScaler()
X_train[numerical_columns] = s_scaler.fit_transform(X_train[numerical_columns])
X_test[numerical_columns] = s_scaler.transform(X_test[numerical_columns])

  # Random Forest classifier
model= RandomForestClassifier()
model.fit(X_train,y_train)
  # Predict
y_pred = model.predict(X_test)
st.write('The machine learning model that I used is Random Forest Classifier which achieved 93% accuracy')
st.write("accuracy score:", accuracy_score(y_test, y_pred))


  ###########Concolusion###############
st.subheader('conclusion')
st.write('By analysing of data we found that:')
st.write('* USA stand at the first place with the highets number of medals and after that Russia and Germany have taken the second and thirs place respectively. \n* Most popular sports are Ayhletics, Swimming, Rowing and Gymnastics with highest number of medal winners. \n* Although USA has earned their highest number of medals in Athletcs sport, its most successful athletes are in swimming (8 of 10 first place).\n* While USA loves swimming , Russia likes Gymnastics, Germany has penchant for swimming and speed skating and mostly all of the like athletics. \n* Although in 1986 when Olympic started the number of women who participated was 0, it has increased significantly over the years. Women , who were not a very larg part of the Olympics games,now enjoy participating in olympics events. \n* Although the total number of Gold, Silver and Bronze medals is almost equal, in each of them the number of mens medals is three times that of womens.')
