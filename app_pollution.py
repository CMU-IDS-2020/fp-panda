import streamlit as st
import pandas as pd
import altair as alt
import numpy as np
import datetime
from vega_datasets import data
import geopandas as gpd

st.title('Air Pollution Study of United States from 2000 to 2016')


st.write("Air pollution has become a more and more serious issue and it will exacebate climate change. In different studies, it has shown that air pollution could cause different health issues and different air pollutant could be an important simulus for respiratory disease and cardiovascular disease. Air pollution is a mixture of different components, and some of the main pollutants of great concern includes gaseous ozone, nitrogen dioxide, carbon monoxide, and surphur dioxide. The explored dataset contains the recorded the information of four major pollutants for each day from 2000-01-01 to 2016-05-31. Different visualizatiosn are presented to explore the trend of different pollutant over time for different states")

#https://academic.oup.com/eurheartj/article/36/2/83/2293343

def read_data(files):
    total_df=[]
    for f in files:
        df=pd.read_csv(f)
        df=df.iloc[:,1:]
        df['Date Local']=pd.to_datetime(df['Date Local'])
        total_df.append(df)
        # df=df.drop(['State Code','County Code','Site Num','Address','NO2 Units','O3 Units','SO2 Units','CO Units'],axis=1)
    df=pd.concat(total_df, axis=0)
    return df
files=[f'data/US_pollution_2000_2016_{i}.csv' for i in range(0, 10)]
df=read_data(files)

df_state=df[['State','Date Local','NO2 AQI','O3 AQI','SO2 AQI','CO AQI']]
df_state=df_state.dropna().reset_index(drop=True)
# df_state['Date Local']=pd.to_datetime(df_state['Date Local'])
df_state['Year']=pd.to_datetime(df_state['Date Local']).dt.to_period('Y')
df_state['Year_Month']=pd.to_datetime(df_state['Date Local']).dt.to_period('M')

pollutant_list=['NO2', 'O3', 'CO', 'SO2']
state_list=list(df_state.State.value_counts().index)


pollutant_name=st.selectbox(
    'Select a Pollutant ',
     pollutant_list)
pollutant=pollutant_name+' AQI'

states = st.multiselect(
     'Select States ',
     state_list,
     [state_list[0], state_list[1]])

frequency = st.selectbox(
     'Select Plotting Frequency ',
     ['Year','Month', 'Day'])

year_range = st.slider(
     'Select a range of values',
     datetime.datetime(2000, 1, 1),datetime.datetime(2016, 5, 31), (datetime.datetime(2009, 1, 1), datetime.datetime(2014, 3, 1)),format="MM/DD/YYYY")


def filter_state(df_state):
    sub_df=df_state.loc[df_state.State.isin(states)]
    return sub_df
def filter_pollutant(sub_df):
    sub_df=sub_df.loc[:,['State', 'Date Local','Year', 'Year_Month', pollutant]]
    return sub_df

def filter_freq(sub_df):
    if frequency=='Year':
        sub_df=sub_df.groupby(['State','Year']).agg('mean').reset_index()
        sub_df['Year']=pd.PeriodIndex(sub_df.Year, freq='Y').to_timestamp()
    elif frequency=='Month':
        sub_df=sub_df.groupby(['State','Year_Month']).agg('mean').reset_index()
        sub_df['Year_Month']=pd.PeriodIndex(sub_df.Year_Month, freq='M').to_timestamp()
#         sub_df=sub_df.drop('Year', axis=1)
    else:
        sub_df=sub_df.loc[:,['State', 'Date Local', pollutant]]
    sub_df.columns=['State', 'Time', 'AQI']
    return sub_df
    
def filter_time(sub_df):
    sub_df=sub_df[(sub_df['Time']>year_range[0]) & (sub_df['Time']<year_range[1])] 
    return sub_df

sub_df=filter_state(df_state)
sub_df=filter_pollutant(sub_df)
sub_df=filter_freq(sub_df)
sub_df=filter_time(sub_df)

# sub_df=df_state.melt(id_vars=['State', 'Date Local'], var_name='Pollutant_Type', value_name='AQI')

# graph 1
st.write(f'Air Quality Index (AQI) for {pollutant_name} over time')
scatter_chart=st.altair_chart(
    alt.Chart(sub_df,height=350,width=700).mark_line().encode(
        x='Time:T',
        y='AQI',
        color='State'
    ).interactive()
)

# graph 2

def filter_time(sub_df):
    sub_df=sub_df[(sub_df['Date Local']>year_range[0]) & (sub_df['Date Local']<year_range[1])] 
    return sub_df

source=filter_time(df_state)
source=source.groupby('State')['NO2 AQI', 'O3 AQI','SO2 AQI','CO AQI'].agg('mean').reset_index()
source.columns=['state', 'NO2 AQI', 'O3 AQI','SO2 AQI','CO AQI']

cur=pd.read_csv('https://vega.github.io/vega-datasets/data/population_engineers_hurricanes.csv')
source=cur[['state','id']].merge(source, left_on='state', right_on='state', how='left')
source=source.replace(np.nan, 0)
# source.drop('state', axis=1, inplace=True)
# variable_list=['NO2 AQI', 'O3 AQI', 'CO AQI', 'SO2 AQI']


highlight = alt.selection_single(on='mouseover', fields=['state'], empty='none')
states=alt.topo_feature(data.us_10m.url, 'states')
state_map=alt.Chart(states).mark_geoshape().encode(
    color=alt.condition(highlight, alt.value('yellow'), f'{pollutant}:Q'),
    tooltip=['state:N',f'{pollutant}:Q'],
).transform_lookup(
    lookup='id',
    from_=alt.LookupData(source, 'id', [pollutant, 'state'])
).add_selection(highlight).project(
    type='albersUsa'
).properties(
    width=900,
    height=600
)
state_map

## graph3
st.write('Now we could further investigate the cities with highest pollutants or cities with lowest pollutants')


pollutant_name2=st.selectbox(
    'Select a Pollutant ',
     pollutant_list, key='pollutant2')

year_range2 = st.slider(
     'Select start day and end day ',
     datetime.datetime(2000, 1, 1),datetime.datetime(2016, 5, 31), (datetime.datetime(2009, 1, 1), datetime.datetime(2014, 3, 1)),format="MM/DD/YYYY")

region_type=st.selectbox(
    'Check Max Value Hour at level ',
     ['State', 'County','City'], key='region_type')

city_type=st.selectbox(
    f'Select {region_type} with highest/lowest mean of concentration pollutants ',
     ['Highest', 'Lowest'])

city_num=st.selectbox(
    f'Select top N {region_type} with highest/lowest mean of concentration pollutants ',
     list(range(1, 144)))

def filter_time2(df):
    df=df[(df['Date Local']>year_range2[0]) & (df['Date Local']<year_range2[1])] 
    return df

st.write(f'Top {city_num} {region_type} with highest mean of concentration of {pollutant_name2} within a day')

time_df=filter_time2(df)
var_name=pollutant_name2+' Mean'
if city_type=='Highest':
    top_cities=time_df.groupby(region_type)[var_name].agg('mean').sort_values(ascending=False)[:int(city_num)].index
else:
    top_cities=time_df.groupby(region_type)[var_name].agg('mean').sort_values(ascending=True)[:int(city_num)].index
    
city_df=time_df.loc[time_df[region_type].isin(top_cities)]
current_var_list=[pollutant_name2 + i for i in [' Mean', ' 1st Max Value', ' AQI']]
city_df=city_df[[region_type]+current_var_list].groupby(region_type).agg('mean').reset_index()
city_df=city_df.sort_values(by=current_var_list[1]).reset_index(drop=True)
city_df
city_df=city_df.melt(id_vars=region_type, var_name='Measurements', value_name='Mean value')

scatter_chart=st.altair_chart(
    alt.Chart(city_df).mark_bar(opacity=0.5).encode(
        x=alt.X('Mean value:Q', stack=None),
        y=alt.Y(region_type, sort=None),
        color="Measurements",
    ).properties(
        width=600,
        height=500
    ).interactive()
)


# graph 5:
st.write("We could now turn to the counts of hours in a day when the maximum pollutant concentration was recorded in a given day for a given region. ")
year_range3 = st.slider(
     'Select start day and end day ',
     datetime.datetime(2000, 1, 1),datetime.datetime(2016, 5, 31), (datetime.datetime(2009, 1, 1), datetime.datetime(2014, 3, 1)),format="MM/DD/YYYY", key='yearRange3')

def filter_time3(df):
    df=df[(df['Date Local']>year_range3[0]) & (df['Date Local']<year_range3[1])] 
    return df

time_df=filter_time3(df)

region_type=st.selectbox(
    'Check Max Value Hour at level ',
     ['State', 'County','City'])

if region_type=='State':
    region_name=st.selectbox(
    f'Select a {region_type} ',
     list(df.State.value_counts().index))
elif region_type=='County':
    region_name=st.selectbox(
    f'Select a {region_type} ',
     list(df.County.value_counts().index))
else:
    region_name=st.selectbox(
    f'Select a {region_type} ',
     list(df.City.value_counts().index))
    
max_hour_list=['CO 1st Max Hour', 'O3 1st Max Hour', 'SO2 1st Max Hour','NO2 1st Max Hour']
hour_df=df[[region_type]+max_hour_list].melt(id_vars=region_type, var_name='Pollutant Type', value_name='Max Hour Count') 
hour_df=hour_df.loc[hour_df[region_type]==region_name]
hour_df2=hour_df.groupby([region_type, 'Pollutant Type','Max Hour Count']).size().unstack(fill_value=0)
hour_df2
hour_df=hour_df.groupby([region_type, 'Pollutant Type','Max Hour Count']).size().reset_index()
hour_df.columns=[region_type,'Pollutant Type', 'Hour in a day', 'Max Hour Count']

st.write(f'The count of hours when the maximum pollutants concentration was recorded in a given day for {region_name}')
scatter_chart=st.altair_chart(
alt.Chart(hour_df).mark_bar().encode(
    x='Hour in a day',
    y='Max Hour Count',
    color='Pollutant Type:N',
    row="Pollutant Type:N"
).properties(
    height=100
)
)
