import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import seaborn as sns
import streamlit as st
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor

import helper
import preprocessor

df = pd.read_csv('athlete_events.csv')
region_df = pd.read_csv('noc_regions.csv')

df = preprocessor.preprocess(df,region_df)

st.sidebar.title("Olympics Analysis")
st.sidebar.image('https://e7.pngegg.com/pngimages/1020/402/png-clipart-2024-summer-olympics-brand-circle-area-olympic-rings-olympics-logo-text-sport.png')
user_menu = st.sidebar.radio(
    'Select an Option',
    ('Medal Count','Overall Insights','Country-wise Insights','Athletes Insights','Multiple Regression', 'Decision Tree Regression', 'Gradient Boosting')
)

if user_menu == 'Medal Count':
    st.sidebar.header("Medal Count")
    years,country = helper.country_year_list(df)

    selected_year = st.sidebar.selectbox("Select Year",years)
    selected_country = st.sidebar.selectbox("Select Country", country)

    medal_tally = helper.fetch_medal_tally(df,selected_year,selected_country)
    if selected_year == 'Overall' and selected_country == 'Overall':
        st.title("Overall Count")
    if selected_year != 'Overall' and selected_country == 'Overall':
        st.title("Medal Tally in " + str(selected_year) + " Olympics")
    if selected_year == 'Overall' and selected_country != 'Overall':
        st.title(selected_country + " overall performance")
    if selected_year != 'Overall' and selected_country != 'Overall':
        st.title(selected_country + " performance in " + str(selected_year) + " Olympics")
    st.table(medal_tally)

if user_menu == 'Overall Insights':
    editions = df['Year'].unique().shape[0] - 1
    cities = df['City'].unique().shape[0]
    sports = df['Sport'].unique().shape[0]
    events = df['Event'].unique().shape[0]
    athletes = df['Name'].unique().shape[0]
    nations = df['region'].unique().shape[0]

    st.title("Top Statistics")
    col1,col2,col3 = st.columns(3)
    with col1:
        st.header("Editions")
        st.title(editions)
    with col2:
        st.header("Hosts")
        st.title(cities)
    with col3:
        st.header("Sports")
        st.title(sports)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.header("Events")
        st.title(events)
    with col2:
        st.header("Nations")
        st.title(nations)
    with col3:
        st.header("Athletes")
        st.title(athletes)
    
    df_selected = df[['Year', 'region']]

    nations_over_time = df_selected.groupby(['Year', 'region']).size().reset_index(name='Count')
# Plotting the line chart
    fig = px.line(nations_over_time, x='Year', y='Count', color='region')
    st.title("Participating Nations over the years")
    st.plotly_chart(fig)

    df = df.rename(columns={'NOC': 'region', 'Year': 'Edition'})
# Count the occurrences of each event for each edition
    events_over_time = df.groupby(['Edition', 'Event']).size().reset_index(name='Count')
# Plotting the line chart
    fig = px.line(events_over_time, x='Edition', y='Count', color='Event')
    st.title("Events over the years")
    st.plotly_chart(fig)

    athlete_over_time = helper.data_over_time(df, 'Name')
    fig = px.line(athlete_over_time, x="Edition", y="Name")
    st.title("Athletes over the years")
    st.plotly_chart(fig)

    # st.title("No. of Events over time(Every Sport)")
    # fig,ax = plt.subplots(figsize=(20,20))
    # x = df.drop_duplicates(['Year', 'Sport', 'Event'])
    # ax = sns.heatmap(x.pivot_table(index='Sport', columns='Year', values='Event', aggfunc='count').fillna(0).astype('int'),
    #             annot=True)
    # st.pyplot(fig)

    # st.title("Most successful Athletes")
    # sport_list = df['Sport'].unique().tolist()
    # sport_list.sort()
    # sport_list.insert(0,'Overall')

    # selected_sport = st.selectbox('Select a Sport',sport_list)
    # x = helper.most_successful(df,selected_sport)
    # st.table(x)

if user_menu == 'Country-wise Insights':

    st.sidebar.title('Country-wise Insights')

    country_list = df['region'].dropna().unique().tolist()
    country_list.sort()

    selected_country = st.sidebar.selectbox('Select a Country',country_list)

    country_df = helper.yearwise_medal_tally(df,selected_country)
    fig = px.line(country_df, x="Year", y="Medal")
    st.title(selected_country + " Medal Tally over the years")
    st.plotly_chart(fig)

    st.title(selected_country + " excels in the following sports")
    pt = helper.country_event_heatmap(df,selected_country)
    fig, ax = plt.subplots(figsize=(20, 20))
    ax = sns.heatmap(pt,annot=True)
    st.pyplot(fig)

    st.title("Top 10 athletes of " + selected_country)
    top10_df = helper.most_successful_countrywise(df,selected_country)
    st.table(top10_df)

if user_menu == 'Athletes Insights':
    athlete_df = df.drop_duplicates(subset=['Name', 'region'])

    x1 = athlete_df['Age'].dropna()
    x2 = athlete_df[athlete_df['Medal'] == 'Gold']['Age'].dropna()
    x3 = athlete_df[athlete_df['Medal'] == 'Silver']['Age'].dropna()
    x4 = athlete_df[athlete_df['Medal'] == 'Bronze']['Age'].dropna()

    fig = ff.create_distplot([x1, x2, x3, x4], ['Overall Age', 'Gold Medalist', 'Silver Medalist', 'Bronze Medalist'],show_hist=False, show_rug=False)
    fig.update_layout(autosize=False,width=1000,height=600)
    st.title("Distribution of Age")
    st.plotly_chart(fig)

    x = []
    name = []
    famous_sports = ['Basketball', 'Judo', 'Football', 'Tug-Of-War', 'Athletics',
                     'Swimming', 'Badminton', 'Sailing', 'Gymnastics',
                     'Art Competitions', 'Handball', 'Weightlifting', 'Wrestling',
                     'Water Polo', 'Hockey', 'Rowing', 'Fencing',
                     'Shooting', 'Boxing', 'Taekwondo', 'Cycling', 'Diving', 'Canoeing',
                     'Tennis', 'Golf', 'Softball', 'Archery',
                     'Volleyball', 'Synchronized Swimming', 'Table Tennis', 'Baseball',
                     'Rhythmic Gymnastics', 'Rugby Sevens',
                     'Beach Volleyball', 'Triathlon', 'Rugby', 'Polo', 'Ice Hockey']
    for sport in famous_sports:
        temp_df = athlete_df[athlete_df['Sport'] == sport]
        x.append(temp_df[temp_df['Medal'] == 'Gold']['Age'].dropna())
        name.append(sport)

    fig = ff.create_distplot(x, name, show_hist=False, show_rug=False)
    fig.update_layout(autosize=False, width=1000, height=600)
    st.title("Distribution of Age wrt Sports(Gold Medalist)")
    st.plotly_chart(fig)

    sport_list = df['Sport'].unique().tolist()
    sport_list.sort()
    sport_list.insert(0, 'Overall')

    st.title('Height Vs Weight')
    selected_sport = st.selectbox('Select a Sport', sport_list)
    temp_df = helper.weight_v_height(df,selected_sport)
    fig, ax = plt.subplots()
    sns.scatterplot(data=temp_df, x='Weight', y='Height', hue='Medal', style='Sex', s=60, ax=ax)
    st.pyplot(fig)

    st.title("Men Vs Women Participation Over the Years")
    final = helper.men_vs_women(df)
    fig = px.line(final, x="Year", y=["Male", "Female"])
    fig.update_layout(autosize=False, width=1000, height=600)
    st.plotly_chart(fig)


if user_menu == 'Multiple Regression':
    # Read data from CSV file into a dataframe
    df = pd.read_csv('data.csv')

    # Perform multiple regression on the data
    model = LinearRegression()
    X = df[['Athletes', 'Events']]
    y = df['Cost']
    model.fit(X, y)

    # Print the coefficients of the linear regression model
    print('Intercept:', model.intercept_)
    print('Coefficients:', model.coef_)

    # Get user-defined values for number of athletes and events
    num_athletes = st.number_input("Number of Athletes", value=0, step=1)
    num_events = st.number_input("Number of Events", value=0, step=1)

    # Predict the amount spent on user-defined number of players and events
    prediction = model.predict([[num_athletes, num_events]])
    print('Predicted amount spent on', num_athletes, 'players and', num_events, 'events:', prediction)

    # Calculate mean squared error
    y_pred = model.predict(X)
    mse = mean_squared_error(y, y_pred)
    rmse = np.sqrt(mse)

    # Display the mean squared error and root mean squared error
    st.text("Mean Squared Error: " + str(mse))
    st.text("Root Mean Squared Error: " + str(rmse))

    # Plot the data and regression plane
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(df['Athletes'], df['Events'], df['Cost'], color='b')
    x_surf, y_surf = np.meshgrid(np.linspace(df['Athletes'].min(), df['Athletes'].max(), 10),
                                 np.linspace(df['Events'].min(), df['Events'].max(), 10))
    z_surf = model.intercept_ + model.coef_[0] * x_surf + model.coef_[1] * y_surf
    ax.plot_surface(x_surf, y_surf, z_surf, color='r', alpha=0.2)
    ax.set_xlabel('Number of Athletes')
    ax.set_ylabel('Number of Events')
    ax.set_zlabel('Amount Spent')
    ax.set_title('Multiple Regression between Number of Players, Number of Events, and Amount Spent')

    st.title("Prediction")
    st.header("Total amount spent on {} athletes and {} events in billion:".format(num_athletes, num_events))
    pr = "{:.2f}".format(prediction[0])
    st.header(pr)
    st.pyplot(fig)



if user_menu == 'Decision Tree Regression':
    # Read data from CSV file into a dataframe
    df = pd.read_csv('data.csv')

    # Perform decision tree regression on the data
    model = DecisionTreeRegressor()
    X = df[['Athletes', 'Events']]
    y = df['Cost']
    model.fit(X, y)

    # Get user-defined values for number of athletes and events
    num_athletes = st.number_input("Number of Athletes", value=0, step=1)
    num_events = st.number_input("Number of Events", value=0, step=1)

    # Predict the amount spent on user-defined number of players and events
    prediction = model.predict([[num_athletes, num_events]])
    print('Predicted amount spent on', num_athletes, 'players and', num_events, 'events:', prediction)

    # Calculate mean squared error
    y_pred = model.predict(X)
    mse = mean_squared_error(y, y_pred)
    rmse = np.sqrt(mse)

    # Display the mean squared error and root mean squared error
    st.text("Mean Squared Error: " + str(mse))
    st.text("Root Mean Squared Error: " + str(rmse))

    # Plot the data and regression plane
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(df['Athletes'], df['Events'], df['Cost'], color='b')
    x_surf, y_surf = np.meshgrid(np.linspace(df['Athletes'].min(), df['Athletes'].max(), 10),
                                 np.linspace(df['Events'].min(), df['Events'].max(), 10))
    z_surf = model.predict(np.array([x_surf.flatten(), y_surf.flatten()]).T).reshape(x_surf.shape)
    ax.plot_surface(x_surf, y_surf, z_surf, color='r', alpha=0.2)
    ax.set_xlabel('Number of Players')
    ax.set_ylabel('Number of Events')
    ax.set_zlabel('Amount Spent')
    ax.set_title('Decision Tree Regression between Number of Players, Number of Events, and Amount Spent')

    st.title("Prediction")
    st.header("Total amount spent on {} players and {} events in billion:".format(num_athletes, num_events))
    pr = "{:.2f}".format(prediction[0])
    st.header(pr)
    st.pyplot(fig)


if user_menu == 'Gradient Boosting':
    # Read data from CSV files into dataframes
    df = pd.read_csv('data.csv')
    gdp_df = pd.read_csv('gdp.csv', encoding='utf-8')

    # Perform gradient boosting regression on the data
    model = GradientBoostingRegressor()
    X = df[['Athletes', 'Events']]
    y = df['Cost']
    model.fit(X, y)

    # Get user-defined values for number of athletes and events
    num_athletes = st.number_input("Number of Athletes", value=0, step=1)
    num_events = st.number_input("Number of Events", value=0, step=1)

    # Predict the amount spent on user-defined number of players and events
    prediction = model.predict([[num_athletes, num_events]])
    print('Predicted amount spent on', num_athletes, 'players and', num_events, 'events:', prediction)

    # Calculate mean squared error
    y_pred = model.predict(X)
    mse = mean_squared_error(y, y_pred)
    rmse = np.sqrt(mse)

    # Display the mean squared error and root mean squared error
    st.text("Mean Squared Error: " + str(mse))
    st.text("Root Mean Squared Error: " + str(rmse))

    # Plot the data and regression plane
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(df['Athletes'], df['Events'], df['Cost'], color='b')
    x_surf, y_surf = np.meshgrid(np.linspace(df['Athletes'].min(), df['Athletes'].max(), 10),
                                 np.linspace(df['Events'].min(), df['Events'].max(), 10))
    z_surf = model.predict(np.array([x_surf.flatten(), y_surf.flatten()]).T).reshape(x_surf.shape)
    ax.plot_surface(x_surf, y_surf, z_surf, color='r', alpha=0.2)
    ax.set_xlabel('Number of Players')
    ax.set_ylabel('Number of Events')
    ax.set_zlabel('Amount Spent')
    ax.set_title('Gradient Boosting Regression between Number of Players, Number of Events, and Amount Spent')

    st.title("Prediction")
    st.header("Total amount spent on {} players and {} events in billion:".format(num_athletes, num_events))
    pr = "{:.2f}".format(prediction[0])
    st.header(pr)

    # Check if GDP is sufficient to host the Olympics
    selected_country = st.sidebar.selectbox('Select a Country', gdp_df['country'])
    selected_gdp = gdp_df.loc[gdp_df['country'] == selected_country, 'GDP'].values[0]

    if float(pr) < 0.01 * selected_gdp:
        st.success(f"{selected_country} can potentially host the Olympics based on its GDP.")
    else:
        st.error(f"{selected_country} cannot host the Olympics based on its GDP.")

    # Display the GDP of the selected country in the sidebar
    st.sidebar.header("GDP of {} in billion:".format(selected_country))
    st.sidebar.write("{:.2f}".format(selected_gdp))

    st.pyplot(fig)

