import pandas as pd
import dash
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
from dash import dcc
from dash import html
import plotly.express as px
import os
import numpy as np
import country_converter as coco
import plotly.graph_objects as go
from sklearn.impute import SimpleImputer 


df1 = pd.read_csv(r"C:\Users\Rudik\Documents\Python files\Life Expectancy Data.csv")
print(df1.head())

app = dash.Dash(__name__)

df = pd.read_csv(r"C:\Users\Rudik\Documents\Python files\Life Expectancy Data.csv")

#####################################################################################################
#                                   DATA CLEANING                                                   #
#####################################################################################################
#df.drop(['Hepatitis B', 'GDP', 'Population'], axis=1, inplace=True)
df[['percentage expenditure','Income composition of resources','Schooling']].replace(0, np.nan, inplace=True)
# Columns to handle missing values 
columns_to_impute = df.select_dtypes(exclude='object').columns
# columns_to_impute = ['percentage expenditure','Total expenditure', 'Alcohol', 'Income composition of resources', 'Schooling']

# Initialize SimpleImputer with strategy='mean'
imp = SimpleImputer(strategy='mean')

# Impute missing values for specified columns
df[columns_to_impute] = imp.fit_transform(df[columns_to_impute])

from sklearn.impute import SimpleImputer

# Select numerical columns
num_cols = df.select_dtypes(exclude='object').columns

# Apply SimpleImputer for numerical columns
imp = SimpleImputer(strategy='mean')
df[num_cols] = imp.fit_transform(df[num_cols])

# change the status to developed for canada and france
df.loc[df['Country'] == 'Canada', 'Status'] = 'Developed'
df.loc[df['Country'] == 'France', 'Status'] = 'Developed'

# Job titles
nations_options = [{'label': i, 'value': i} for i in df['Status'].unique()]
nations_options.append({'label': 'All Nations', 'value': 'all'})


country_options = [{'label': i, 'value': i} for i in df['Country'].unique()]
country_options.append({'label': 'All Countries', 'value': 'all'})

year_options = [{'label': i, 'value': i} for i in df['Year'].unique()]
# year_options.append({'label': 'All Years', 'value': 'all'})


#####################################################################################################
#                                   APP LAYOUT                                                      #
#####################################################################################################

app.layout = html.Div([

                html.Div(children=[
                    
                    html.Div(children=[ 
                        html.H1(
                                "Filters",
                                style={"font-weight":"bold", "font-size":"30px","margin-left": "14px", "margin-right": "14px","margin-top": "14px","margin-bottom": "28px"}
                            ),
                    html.H2(
                                "Select Nation",
                                style={"font-weight":"bold", "font-size":"24px","margin-left": "14px", "margin-right": "14px","margin-top": "14px","margin-bottom": "28px"}
                            ),
                        dcc.Dropdown(
                                id='nations_test',
                                options=nations_options,
                                value='all',
                                clearable=False,
                                style={"margin-left": "6px", "margin-right": "6px", "width":"98.5%","margin-top": "14px","margin-bottom": "28px","font-size":"16px"}
                            ),
                        html.H2(
                                "Select Country",
                                style={"font-weight":"bold", "font-size":"24px","margin-left": "14px", "margin-right": "14px","margin-top": "14px","margin-bottom": "28px"}
                            ),
                        dcc.Dropdown(
                                id='country_test',
                                options=country_options,
                                value='all',
                                clearable=False,
                                style={"margin-top": "6px","margin-bottom": "28px", "margin-left": "6px", "margin-right": "6px", "width":"98.5%","font-size":"16px"}
                            ),
                        dcc.Markdown("### Select Year Range", style={"font-weight":"bold", "font-size":"24px","margin-left": "14px", "margin-right": "14px","margin-top": "14px","margin-bottom": "16px"}),
                        dcc.RangeSlider(2000,2015,1, value=[2000,2015],id='year_slider',marks={i: '{}'.format(i) for i in range(2000,2016,1)},
                            updatemode='drag',included=False
                            )],
                        # Div style for child #1
                        style={"backgroundColor": "#FFFFFF", "width":"49.5%","display":"inline-block","height":"300px",
                        'margin-left': '6px', 'margin-right': '2px', "vertical-align":"top",'margin-top': '8px',}),# "border-radius":"10px"}),

                    html.Div(children=[
                        dcc.Graph(
                                id="boxplot",
                                #style={"width":"753px"}
                            )],
                        # Div style for child #2
                        style={"width":"49.5%","display":"inline-block", 'margin-right': '4px', 'margin-left': '2px','margin-top': '4px'})
                ]),

                html.Div(children=[
                    dcc.Graph(
                        id="worldplot",
                        style={'width':'49.5%','display': 'inline-block','margin-left': '6px', 'margin-right': '2px'}
                    ),
                    dcc.Graph(
                        id="correlationplot",
                        style={'width':'49.5%','display': 'inline-block','margin-right': '2px', 'margin-left': '8px','margin-top': '0px'}
                    )
                ])

            # Div style for whole page    
            ], style={"backgroundColor": "#F2F2F2"})
      

#####################################################################################################
#                                   CALL BACK AND CHARTS                                            #
#####################################################################################################

@app.callback(
    Output(component_id="boxplot", component_property="figure"), 
    Output(component_id="correlationplot",component_property="figure"),
    Output(component_id="worldplot", component_property="figure"),
    Input(component_id='nations_test', component_property='value'), #
    Input(component_id='country_test', component_property='value'), #
    Input(component_id='year_slider', component_property='value'),
)
def generate_chart(nations_test, country_test,year_slider):
    
    dff = df.copy()
    timedf = df.copy()
    timedf = dff.groupby('Year')['Life expectancy '].mean()
    print(year_slider)
    # Filters
    if nations_test != 'all':
        dff = dff[dff['Status'] == nations_test]
        timedf = dff.groupby('Year')['Life expectancy '].mean()
    if country_test != 'all':
        dff = dff[dff['Country'] == country_test]
        timedf = dff.groupby('Year')['Life expectancy '].mean()
    
    dff = dff[dff['Year'] <= year_slider[1]]
    dff = dff[dff['Year'] >= year_slider[0]]
    
    # tidf = tidf[tidf['Year'] == year_slider]

    
    print(dff.Year.unique())
    print(dff.isnull().sum())
    print(dff.select_dtypes(exclude='object').corr())
    # Boxplot graph
    
    fig_boxplot = px.line(
    x=timedf.index,
    y=timedf.values,
    title='Time Series Plot of Life Expectancy over the years',
    labels={'Year': 'Year', 'Life expectancy ': 'Life Expectancy'},
)

    print(dff.head(10))
   # Correlation graph
    fig_correlationplot = px.imshow(
        
        dff.select_dtypes(exclude='object').corr().fillna(0), # Using dff instead of selecting types excluding 'object'
        color_continuous_scale='RdBu',
        range_color=(-0.70, 1)
    )

    # Update layout for better visibility of all variable labels
    fig_correlationplot.update_layout(
        title='Correlation Matrix of Variables',
        height=800,  # Adjust height as needed
        width=1200,  # Adjust width as needed
        xaxis=dict(tickangle=-45),
        yaxis=dict(tickangle=30),
        margin=dict(l=100, r=200, b=100, t=150, pad=4)  # Margins to adjust for better label visibility wherr r is right, l is left, b is bottom, t is top  
    )

    fig_worldplot = px.choropleth(dff, locations='Country', locationmode='country names', color='Life expectancy ',color_continuous_scale='ylgn',range_color=(40, 80))
    fig_worldplot.update_layout(
    height=800,  # Adjust height as needed
    width=1200  # Adjust width as needed
    )
    fig_worldplot.update_geos(
    showland=True, landcolor="LightGrey"
    )                  
                          
    return fig_boxplot,fig_correlationplot, fig_worldplot
    
if __name__ == '__main__':  
    app.run_server(debug=False, port=8080)


