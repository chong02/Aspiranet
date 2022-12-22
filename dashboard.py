# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import *
import geopandas as gpd
import folium
from plotly import graph_objects as go
from plotly.subplots import make_subplots
import re
from scipy import stats

## Data cleaning functions ##
ca_counties = ['alameda', 'alpine', 'amador', 'butte', 'calaveras',
               'colusa', 'contra costa', 'del norte', 'el dorado', 'fresno',
               'glenn', 'humboldt', 'imperial', 'inyo', 'kern', 
               'kings', 'lake', 'lassen', 'los angeles', 'madera',
               'marin', 'mariposa', 'mendocino', 'merced', 'modoc',
               'mono', 'monterey', 'napa', 'nevada', 'orange',
               'placer', 'plumas', 'riverside', 'sacramento', 'san benito',
               'san bernardino', 'san diego', 'san francisco', 'san joaquin', 'san luis obispo',
               'san mateo', 'santa barbara', 'santa clara', 'santa cruz', 'shasta',
               'sierra', 'siskiyou', 'solano', 'sonoma', 'stanislaus', 
               'sutter', 'tehama', 'trinity', 'tulare', 'tuolumne', 
               'ventura', 'yolo', 'yuba']

def clean_county_col(df, county_col):
    '''
    Given DataFrame df of shape (r, c) and string county_col, returns a Series
    counties of shape (r, 1) with extracted county names
    -----
    Input:
    
    df (DataFrame) - DataFrame with CountyOfService column to clean
    
    county_col (string) - Name of column in df with county names
    -----
    Output:
    
    counties (Series) - Series with extracted county names
    '''
    df = df.copy()
    df[county_col] = df[county_col].str.lower()
    
    # replace abbreviations
    df[county_col] = df[county_col].str.replace(r'stan', 'stanislaus', regex=True)
    df[county_col] = df[county_col].str.replace(r'l.a.', 'los angeles', regex=True)
    df[county_col] = df[county_col].str.replace(r'san fran', 'san francisco', regex=True)
    
    # map district to county
    district_to_county_dict = { 
        '|'.join(['oxnard','simi valley','moorpark','conejo valley', 'briggs',
        'santa paula','ojai','hueneme','fillmore','oak park']) : 'ventura',
        '|'.join(['las virgenes','palmdale']) : 'los angeles',
        'ocean view' : 'orange',
        'kaiser': 'sacramento',
        'berkeley' : 'alameda',
        'csma' : 'santa clara', #looks like it serves both santa clara and san mateo?
        '|'.join(['strtp','excell center']) : 'stanislaus' #UNSURE
        # what is 'pcit'?
    }
    df.replace(to_replace=district_to_county_dict, inplace=True, regex=True)
    
    p = '({})'.format('|'.join(ca_counties))
    df['cleaned_county'] = df[county_col].str.extract(p, expand=False).fillna('unknown')
    counties = df['cleaned_county']
    #only return col, not df because other cols have been mutated
    return counties

def convert_to_datetime(string, format='%m/%d/%Y'):
    '''
    Function that takes a string in format MM/DD/YYYY and converts it to
    a Python datetime object
    -----
    Input:
    
    string (str) - String object to convert
    -----
    Output:
    
    time_obj (datetime) - datetime object
    '''
    #format = '%m/%d/%Y'
    time_obj = datetime.strptime(string, format)
    return time_obj

## Read in data ##
demographics = pd.read_csv('Data/Demographics.csv') # Data must exist at designated file path
incidents = pd.read_csv('Data/Incidents.csv') # Data must exist at designated file path

# Clean county names
demographics['County'] = clean_county_col(demographics, 'CountyofService')
demographics['County Upper'] = [' '.join([word.capitalize() for word in re.split('\s', county_name)])
                                for county_name in demographics['County']]
demographics = demographics.drop('County', axis=1) \
                           .rename(columns={'County Upper': 'County'})
incidents['County'] = clean_county_col(incidents, 'CountyOService')
incidents['County Upper'] = [' '.join([word.capitalize() for word in re.split('\s', county_name)])
                             for county_name in incidents['County']]
incidents = incidents.drop('County', axis=1) \
                     .rename(columns={'County Upper': 'County'})

# Clean IncidentDate
incidents['IncidentDatetime'] = [convert_to_datetime(object) for object in incidents['IncidentDate'].values]
incidents['Month'] = [date.month for date in incidents['IncidentDatetime']]
incidents['Year'] = [date.year for date in incidents['IncidentDatetime']]

# Reorder incidents dataframe
incidents = incidents[['IncidentDatetime', 'IncidentDate', 'Month', 'Year',
                       'IncidentTypeCode', 'IncidentType', 'SubTypeCode',
                       'IncidentSubType', 'OptionsNumber', 'AgeAtIncident',
                       'Gender', 'Ethnicity', 'PrimaryLanguage',
                       'Religion', 'ServiceDivision','County']]

# Counties present in data
counties_present_lower = demographics[['County']].query('County != "Unknown"')['County'].unique()
counties_present = [' '.join([word.capitalize() for word in re.split('\s', county_name)])
                    for county_name in counties_present_lower]

# Create county_seats dataframe
county_seats_dict = {'County': ['Alameda', 'Alpine', 'Amador',
                                'Butte', 'Calaveras', 'Colusa',
                                'Contra Costa', 'Del Norte', 'El Dorado',
                                'Fresno', 'Glenn', 'Humboldt',
                                'Imperial', 'Inyo', 'Kern',
                                'Kings', 'Lake', 'Lassen',
                                'Los Angeles', 'Madera', 'Marin',
                                'Mariposa', 'Mendocino', 'Merced',
                                'Modoc', 'Mono', 'Monterey',
                                'Napa', 'Nevada', 'Orange',
                                'Placer', 'Plumas', 'Riverside',
                                'Sacramento', 'San Benito', 'San Bernardino',
                                'San Diego', 'San Francisco', 'San Joaquin',
                                'San Luis Obispo', 'San Mateo', 'Santa Barbara',
                                'Santa Clara', 'Santa Cruz', 'Shasta',
                                'Sierra', 'Siskiyou', 'Solano',
                                'Sonoma', 'Stanislaus', 'Sutter',
                                'Tehama', 'Trinity', 'Tulare',
                                'Tuolomne', 'Ventura', 'Yolo',
                                'Yuba'],
                     'County Seats': ['Oakland', 'Markleeville', 'Jackson',
                                      'Oroville', 'San Andreas', 'Colusa',
                                      'Martinez', 'Crescent City', 'Placerville',
                                      'Fresno', 'Willows', 'Eureka',
                                      'El Centro', 'Independence', 'Bakersfield',
                                      'Hanford', 'Lakeport', 'Susanville',
                                      'Los Angeles', 'Madera', 'San Rafael',
                                      'Mariposa', 'Ukiah', 'Merced',
                                      'Alturas', 'Bridgeport', 'Salinas',
                                      'Napa', 'Nevada City', 'Santa Ana',
                                      'Auburn', 'Quincy', 'Riverside',
                                      'Sacramento', 'Hollister', 'San Bernardino',
                                      'San Diego', 'San Francisco', 'Stockton',
                                      'San Luis Obispo', 'Redwood City', 'Santa Barbara',
                                      'San Jose', 'Santa Cruz', 'Redding',
                                      'Downieville', 'Yreka', 'Fairfield',
                                      'Santa Rosa', 'Modesto', 'Yuba City',
                                      'Red Bluff', 'Weaverville', 'Visalia',
                                      'Sonora', 'Ventura', 'Woodland',
                                      'Marysville'],
                     'Latitude': [37.8044, 38.6937, 38.23488,
                                  39.5138, 38.1960, 39.2143,
                                  38.0194, 41.7558, 38.7296,
                                  36.7378, 39.5243, 40.8021,
                                  32.7920, 36.8027, 35.3733,
                                  36.3275, 39.0430, 40.4163,
                                  34.0522, 36.9613, 37.9735,
                                  37.4849, 39.1502, 37.3022,
                                  41.4871, 38.2557, 36.6777,
                                  38.2975, 39.2616, 33.7455,
                                  38.8966, 39.9368, 33.9806,
                                  38.5816, 36.8525, 34.1083,
                                  32.7157, 37.7749, 37.9577,
                                  35.2828, 37.4848, 34.4208,
                                  37.3387, 36.9741, 40.5865,
                                  39.5595, 41.7354, 38.2492,
                                  38.4404, 37.6393, 39.1404,
                                  40.1785, 40.7310, 36.3302,
                                  37.9829, 34.2805, 38.6785,
                                  39.1457],
                     'Longitude': [-122.2712, -119.7797, -120.7741,
                                   -121.5564, -120.6805, -122.0094,
                                   -122.1341, -124.2026, -120.7985,
                                   -119.7871, -122.1936, -124.1637,
                                   -115.5631, -118.2001, -119.0187,
                                   -119.6457, -122.9158, -120.6530,
                                   -118.2437, -120.0607, -122.5311,
                                   -119.9663, -123.2078, -120.4830,
                                   -120.5425, -119.2314, -121.6555,
                                   -122.2869, -121.0161, -117.8677,
                                   -121.0769, -120.9472, -117.3755,
                                   -121.4944, -121.4016, -117.2898,
                                   -117.1611, -122.4194, -121.2908,
                                   -120.6596, -122.2281, -119.6982,
                                   -121.8853, -122.0308, -122.3917,
                                   -120.8277, -122.6345, -122.0405,
                                   -122.7141, -120.9970, -121.6169,
                                   -122.2358, -122.9420, -119.2921,
                                   -120.3822, -119.2945, -121.7733,
                                   -121.5914]}
complete_county_seats = pd.DataFrame(data=county_seats_dict)
complete_county_seats_geo = gpd.GeoDataFrame(complete_county_seats,
                                             geometry=gpd.points_from_xy(complete_county_seats.Latitude, 
                                                                         complete_county_seats.Longitude))
county_seats_geo = complete_county_seats_geo.query('County in @counties_present') \
                                            .reset_index().drop('index', axis=1)

# for looping to generate Plotly plot
figs = {}
county_list = []
html_list = []
upper_county_list = [] # Strictly to be used to create counties_geo dataframe

for county_name in np.sort(counties_present):
    html_county_name = re.sub('\s', '_', county_name) + '_county'
    title_county_name = ' '.join([word.capitalize() for word in re.split('\s', county_name)])
    upper_county_list.append(title_county_name)
    county_list.append(county_name)
    html_list.append(f'{html_county_name}.html')
    county_incidents = incidents.query('County == @county_name')
    county_incident_count = county_incidents.groupby(['Year', 'Month']).agg({'OptionsNumber': 'count',
                                                                         'AgeAtIncident': 'mean'}) \
                                        .rename(columns={'OptionsNumber': 'Incident Count',
                                                         'AgeAtIncident': 'Average Age'}) \
                                        .reset_index() \
                                        [['Month', 'Year', 'Incident Count', 'Average Age']]

    county_incident_count['Month String'] = [str(item) for item in county_incident_count['Month']]
    county_incident_count['Year String'] = [str(item) for item in county_incident_count['Year']]
    county_incident_count['Unique Month String'] = (county_incident_count['Month String'] + '/' +
                                                    county_incident_count['Year String'])

    county_incident_count['Unique Month'] = [convert_to_datetime(item, '%m/%Y') for item in county_incident_count['Unique Month String']]
    county_incident_count = county_incident_count[['Unique Month', 'Incident Count', 'Average Age']]
    county_incident_count = county_incident_count.rename(columns={'Unique Month': 'Month' })
    county_incident_count['HTML 1'] = [f'<b>{title_county_name} County</b><br><br>Month: ' for i in range(county_incident_count.shape[0])]
    county_incident_count['Text Month'] = [(str(month)[:-19]) for month in county_incident_count['Month'].values]
    county_incident_count['HTML 2'] = ['<br>Average Age: ' for i in range(county_incident_count.shape[0])]
    county_incident_count['Text Age'] = [str(age) for age in county_incident_count['Average Age'].values]
    county_incident_count['HTML 3'] = ['<br>Incident Count: ' for i in range(county_incident_count.shape[0])]
    county_incident_count['Text Count'] = [str(count) for count in county_incident_count['Incident Count'].values]
    county_incident_count['Text'] = (county_incident_count['HTML 1'].values +
                                     county_incident_count['Text Month'].values +
                                     county_incident_count['HTML 2'].values +
                                     county_incident_count['Text Age'].values + 
                                     county_incident_count['HTML 3'].values +
                                     county_incident_count['Text Count'].values)
    county_incident_count = county_incident_count[['Month', 'Incident Count', 'Average Age', 'Text']]

    fig = make_subplots(specs=[[{'secondary_y':True}]])

    fig.add_trace(go.Bar(x=county_incident_count['Month'],
                         y=county_incident_count['Incident Count'],
                         name='Incident Count',
                         hovertext=county_incident_count['Text'],
                         marker_color='#003262'),
                  secondary_y=False)

    fig.add_trace(go.Scatter(x=county_incident_count['Month'],
                             y=county_incident_count['Average Age'],
                             name='Average Age',
                             text=county_incident_count['Text'],
                             hoverinfo='text',
                             line=dict(color='#FDB515',
                                       width=3)),
                  secondary_y=True)

    fig.update_layout(hoverlabel_bgcolor='#B9D3B6',
                      font_family='Georgia',
                      title_text=f'Overview: {title_county_name} County',
                      title_font_size=20,
                      title_x=0.5,
                      hoverlabel=dict(font_family='Georgia'),
                      xaxis=dict(tickfont_size=10,
                                 tickangle=270,
                                 showgrid=True,
                                 zeroline=True,
                                 showline=True,
                                 showticklabels=True,
                                 dtick='M1',
                                 tickformat='%b\n%Y'),
                      legend=dict(orientation='h',
                                  xanchor='center',
                                  x=0.72,
                                  y=1),
                      yaxis_title='Number of Incidents',
                      yaxis2_title='Average Age')

    fig.update_xaxes(rangeslider_visible=True,
                     rangeselector=dict(buttons=list([dict(count=1, label='1m', step='month', stepmode='backward'),
                                                      dict(count=6, label='6m', step='month', stepmode='backward'),
                                                      dict(count=1, label='YTD', step='year', stepmode='todate'),
                                                      dict(count=1, label='1y', step='year', stepmode='backward'),
                                                      dict(step='all')])))
    
    figs[html_county_name] = fig
    fig.write_html(f'Plots/{html_county_name}.html')

county_names_df = pd.DataFrame(data=county_list, columns=['County'])
html_list_df = pd.DataFrame(data=html_list, columns=['HTML File'])
plots_df = pd.concat([county_names_df, html_list_df], axis=1)
county_seats_geo = pd.merge(left=county_seats_geo, right=plots_df,
                            on='County')

# Read in shapefile to plot CA counties
county_borders = gpd.read_file('Data/Counties.zip') # Data must exist at designated file path
county_borders['County'] = [name for name in county_borders['NAME'].values]
clean_county_borders = county_borders[county_borders['County'].isin(counties_present)] \
                                                              .reset_index() \
                                                              .drop('index', axis=1)
counties_geo = clean_county_borders[['County', 'geometry']]

## Base layer statistics ##
pd.options.mode.chained_assignment = None # Used to bypass unecessary warnings raised by Pandas
demo_n = demographics.shape[0]
incidents_n = incidents.shape[0]

counties_geo['Proportion of Clients'] = [round(demographics.loc[demographics['County'] == county].shape[0] / demo_n, 4)
                                         if county in demographics['County'].unique() 
                                         else None for county in counties_present]
counties_geo['Proportion of Incidents'] = [round(incidents.loc[incidents['County'] == county].shape[0] / incidents_n, 4)
                                           if county in incidents['County'].unique()
                                           else None for county in counties_present]

first_quartile = np.percentile(counties_geo['Proportion of Incidents'], 25)
second_quartile = np.percentile(counties_geo['Proportion of Incidents'], 50)
third_quartile = np.percentile(counties_geo['Proportion of Incidents'], 75)

counties_geo['Incident Quartile'] = [1 if prop < first_quartile else
                                     2 if prop < second_quartile else
                                     3 if prop < third_quartile
                                     else 4 for prop in counties_geo['Proportion of Incidents']]
counties_geo['Incident Percentile'] = [round(stats.percentileofscore(counties_geo['Proportion of Incidents'],
                                                                     prop),
                                             2)
                                       for prop in counties_geo['Proportion of Incidents'].values]
counties_geo['Largest Division'] = [demographics.loc[demographics['County'] == county] \
                                                .groupby('ServiceDivision').count() \
                                                .sort_values('ServiceDivision', ascending=False) \
                                                .reset_index() \
                                                .iloc[0, 0]
                                    for county in counties_geo['County'].values]
counties_geo['Top Incident Type'] = [incidents.loc[incidents['County'] == county] \
                                              .groupby('IncidentType').count() \
                                              .sort_values('IncidentType', ascending=False) \
                                              .reset_index() \
                                              .iloc[0, 0]
                                     for county in counties_geo['County'].values]

## Recategorize ethnicity into larger race categories ##
# Setting up larger race categories
hisp = ['Hispanic/Latino','Mexican','Central American','South American']
asian = ['Filipino','Hmong','Other Asian','Laotian','Cambodian','Asian Indian','Vietnamese','Japanese','Korean','Chinese']
aian = ['American Indian ', 'Alaskan Native']
nhopi = ['Hawaiian','Other Asian/Pacific Islander','Guamanian','Other Pacific Islander','Samoan']
others = ['No Preference','[Not Entered]']
new_eth = ['White', 'Black', 'Hispanic/Latino', 'Asian', 'AI/AN', 'NH/OPI']

def get_new_eth(eth):
    value = eth
    if "White" in eth:
        value = "White"
    elif "Black" in eth:
        value = "Black"
    elif eth in hisp:
        value = "Hispanic/Latino"
    elif eth in asian:
        value = "Asian"
    elif eth in aian:
        value = "AI/AN"
    elif eth in nhopi:
        value = "NH/OPI"
    elif eth in others:
        value = "Others"
    return value

## Create base map layer ##
base_map = counties_geo.explore(column='Incident Percentile',
                                name='County-Level Profile (Base)')
loc = 'Aspiranet County-Level Dashboard'
title_html = f'<h3 align="center" style="font-size:16px"><b>{loc}</b></h3>'
base_map.get_root().html.add_child(folium.Element(title_html))

overview = folium.FeatureGroup(name='Incident Time Analysis').add_to(base_map)

for i in range(county_seats_geo.shape[0]):
    iframe_html = f'<iframe src="Plots/{county_seats_geo.iloc[i, 5]}" width="850" height="400" frameborder="0">'
    popup = folium.Popup(folium.Html(iframe_html, script=True))
    folium.Marker([county_seats_geo['Latitude'].iloc[i],
                  county_seats_geo['Longitude'].iloc[i]],
                  popup=popup,
                  icon=folium.Icon(icon='home', prefix='fa')) \
          .add_to(overview)

race_breakdown = demographics.copy()
race_breakdown['Race'] = race_breakdown['Ethnicity'].apply(get_new_eth)
race_geo = counties_geo.copy()[['County', 'geometry']]
race_geo['Total Clients'] = [race_breakdown.query('County == @county').shape[0]
                             if county in race_breakdown['County'].unique()
                             else None for county in counties_present]
def add_race_population(df, df2, race):
    df2[f'{race} Population'] = [round(df.query('County == @county & Race == @race').shape[0] / 
                                      df.query('County == @county').shape[0], 4)
                                      if county in df['County'].unique()
                                      else None for county in counties_present]

races_list = ['AI/AN', 'Asian', 'Black',
              'Hispanic/Latino', 'NH/OPI', 'Others',
              'White']
for race in races_list:
    add_race_population(race_breakdown, race_geo, race)
    
race_layer = race_geo.explore(m=base_map,
                              name='Client Population Racial Makeup')

race_breakdown_incidents = incidents.copy()
race_breakdown_incidents['Race'] = race_breakdown_incidents['Ethnicity'].apply(get_new_eth)
race_incidents_geo = counties_geo.copy()[['County', 'geometry']]
race_incidents_geo['Total Incidents'] = [race_breakdown_incidents.query('County == @county').shape[0]
                                         if county in race_breakdown_incidents['County'].unique()
                                         else None for county in counties_present]
for race in races_list:
    add_race_population(race_breakdown_incidents, race_incidents_geo, race)

race_incidents_layer = race_incidents_geo.explore(m=race_layer,
                                                  color='#F1948A',
                                                  name='Incident Population Racial Makeup')

gender_breakdown = demographics.copy()
gender_geo = counties_geo.copy()[['County', 'geometry']]
gender_geo['Total Clients'] = [gender_breakdown.query('County == @county').shape[0]
                               if county in gender_breakdown['County'].unique()
                               else None for county in counties_present]

gender_mapping = {'M': 'Male', 'F': 'Female',
                  ' ': 'Gender Not Determined', 'U':'Gender Unknown'}
def add_gender_population(df, df2, gender):
    gender_full = gender_mapping.get(gender)
    df2[f'{gender_full} Population'] = [round(df.query('County == @county & Gender == @gender').shape[0] / 
                                        df.query('County == @county').shape[0], 4)
                                        if county in df['County'].unique()
                                        else None for county in counties_present]
gender_list = ['M', 'F', 'U', ' ']
for gender in gender_list:
    add_gender_population(gender_breakdown, gender_geo, gender)
    
gender_layer = gender_geo.explore(m=race_incidents_layer,
                                  color='#F5B041',
                                  name='Client Population Sex Makeup')

gender_breakdown_incidents = incidents.copy()
gender_incidents_geo = counties_geo.copy()[['County', 'geometry']]
gender_incidents_geo['Total Incidents'] = [gender_breakdown_incidents.query('County == @county').shape[0]
                                           if county in gender_breakdown_incidents['County'].unique()
                                           else None for county in counties_present]
for gender in gender_list:
    add_gender_population(gender_breakdown_incidents, gender_incidents_geo, gender)

gender_incidents_layer = gender_incidents_geo.explore(m=gender_layer,
                                                      color='#76D7C4',
                                                      name='Incident Population Sex Makeup')

folium.LayerControl().add_to(gender_incidents_layer)

gender_incidents_layer.save('county_level_dashboard.html') # Saves dashboard as designated name
print('Done!')