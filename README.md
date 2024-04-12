# Airbnb-EDA
Airbnb: “Air Bed and Breakfast” A service that lets property owners rent out their spaces to travelers looking for a place to stay. Travelers can rent a space for multiple people to share, a shared space with private rooms, or the entire property for themselves.

#Explore the Data

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import plotly.express as px

from google.colab import drive
drive.mount('/content/drive')

df = pd.read_csv('/content/drive/MyDrive/AB_NYC_2019.csv')
df.head()

df.head(3).T

df.info()

# change float format
pd.options.display.float_format = '{:,.2f}'.format

df.describe()

df.describe(include='object')

# check for missing values
df.isnull().mean() * 100

df.name.sample(5)

df.host_id.nunique()

df.host_id.value_counts().nlargest(10)

df.host_name.sample(5)

# Drop columns that are not useful
df.drop(['id', 'name', 'host_name'], axis=1, inplace=True)

df.sample(3).T

#Pandas Profiling

!pip install ydata-profiling

import pandas as pd
from ydata_profiling import ProfileReport
# Read the data from a csv file 
df = pd.read_csv('/content/drive/MyDrive/AB_NYC_2019.csv')
# Generate the data profiling report 
profile = ProfileReport(df, title='Air bnb Data')
profile.to_file("Air_bnb.html")

profile

df[df.longitude > -74.2].shape

#Handling Missing Values

df.isnull().mean() * 100

df[df.last_review.isnull()].head(10).T

df.reviews_per_month.fillna(0, inplace=True)
df.last_review = pd.to_datetime(df.last_review)

df.last_review.min(), df.last_review.max()

# fill missing values with the minimum date (Just a convention)
df.last_review.fillna(df.last_review.min(), inplace=True)

df.isnull().sum()

# categorical variables
cat_cols = df.select_dtypes(include='object').columns.tolist()
cat_cols

# explore categorical variables
for col in cat_cols:
    print(f'{col}: {df[col].nunique()}')

# explore neighbourhood_group
df.neighbourhood_group.value_counts()

nhg_pct = df.neighbourhood_group.value_counts(normalize = True)
nhg_pct

plt.pie(nhg_pct, labels=nhg_pct.index, autopct='%1.1f%%');

px.pie(nhg_pct, values=nhg_pct.values, names=nhg_pct.index, title='Neighbourhood Group')

df.groupby(['neighbourhood_group', 'room_type'])['price'].mean()

df.groupby(['neighbourhood_group', 'room_type'])['price'].mean().sort_values(ascending=False)

sns.barplot(x='neighbourhood_group', y='price', hue='room_type', data=df, order=df.neighbourhood_group.value_counts().index);

# facetgrid
g = sns.FacetGrid(df, col='room_type', height=4, aspect=1.5, palette='Set1')
g.map(sns.barplot, 'neighbourhood_group', 'price', order=df.neighbourhood_group.value_counts().index);

# facetgrid plotly
fig = px.histogram(df, x='neighbourhood_group', y='price', color='room_type', facet_col='room_type', facet_col_wrap=3,
                category_orders={'neighbourhood_group': df.neighbourhood_group.value_counts().index}, histfunc='avg')
fig.show()

pd.crosstab(index = df.neighbourhood_group, columns = df.room_type, values = df.price, aggfunc='mean')

nh_group_room = pd.pivot_table(df, index='neighbourhood_group', columns='room_type', values='price', aggfunc='mean')
nh_group_room

plt.figure(figsize=(12, 8))
sns.heatmap(nh_group_room, annot=True, fmt='.0f');

The most expensive neighbourhood: Manhattan, then Brooklyn (Entire home/apt or Private room)

The least expensive neighbourhood: Brooklyn (Shared room)

#> In general you can specify the room type  first, if it has high priority, then compare the prices of the neighbourhoods. Or, you can specify the neighbourhood first, if it has high priority, then compares the price of the room types.

# free rooms
df[df.price == 0]

df.neighbourhood.value_counts().nlargest(10)

sns.countplot(y='neighbourhood', data=df, order=df.neighbourhood.value_counts().nlargest(10).index);

df.groupby('neighbourhood')['price'].mean().nlargest(10)

df.groupby(['neighbourhood_group', 'neighbourhood'])['price'].mean().nlargest(10)

data = df.groupby(['neighbourhood_group', 'neighbourhood', 'room_type'])['price'].mean().nlargest(10).reset_index()
data

# Intersted in some city
def get_prices(city, max_price=200, room_type='Entire home/apt'):

    data = df[(df.neighbourhood_group == city) & (df.price < max_price) & (df.room_type == room_type)]
    data = data.groupby(['neighbourhood'])['price'].mean()
    return data

get_prices('Manhattan', 500, 'Private room').nlargest(10)

get_prices('Manhattan', 500, 'Private room').nsmallest(10)

px.histogram(df, x='price', color='room_type')

#Host Analysis

df.host_id.value_counts().nlargest(10)

#Location Analysis

plt.figure(figsize=(12, 8))
sns.scatterplot(x='longitude', y='latitude', data=df, hue='neighbourhood_group', palette='Set1');

px.scatter(df, x='longitude', y='latitude', color='neighbourhood_group' )

df.price.describe()

sns.boxplot(x='price', data=df);

df.price.quantile(0.99)

df[df.price < 800].price.hist(bins=100);

px.scatter(df[df.price>1000], x='longitude', y='latitude', color='neighbourhood_group', size='price', size_max=15, hover_name='neighbourhood')

#Availability Analysis

df.availability_365.value_counts()

#Minimum Nights Analysis

df.minimum_nights.value_counts()

#Reviews Analysis

px.histogram(df, x='reviews_per_month')

px.histogram(df, x='number_of_reviews')

px.box(df, x='neighbourhood_group', y='price')

px.box(df[df.price<1000], x='neighbourhood_group', y='price')

px.box(df, x='room_type', y='price')

px.scatter(df, x='number_of_reviews', y='price')

sns.pairplot(df[['price', 'minimum_nights', 'number_of_reviews', 'reviews_per_month', 'calculated_host_listings_count', 'availability_365']]);

df.sort_values('number_of_reviews', ascending=False).head(10)

#Tasks

#Create a new column called distance that contains the distance between the property and the city center (latitude: 40.7128, longitude: -74.0060).

# New York City Center Coordinates
lat = 40.7128
lon = -74.0060

# create "distance" feature
import geopy.distance

def get_distance(lat1, lon1, lat2=lat, lon2=lon):
    coords_1 = (lat1, lon1)
    coords_2 = (lat2, lon2)
    return geopy.distance.distance(coords_1, coords_2).km

df['distance'] = df.apply(lambda x: get_distance(x.latitude, x.longitude), axis=1)

df.groupby('neighbourhood_group')['distance'].mean()

df.groupby(['neighbourhood_group', 'room_type'])['price'].mean().sort_values(ascending=False)

df.groupby(['neighbourhood_group', 'room_type'])['price'].median().sort_values(ascending=False)

df.price.describe()

df.groupby(['neighbourhood_group', 'neighbourhood', 'room_type'])['price'].mean().nlargest(10).reset_index()

df.groupby(['neighbourhood_group', 'neighbourhood', 'room_type'])['price'].median().nlargest(10).reset_index()

df2= df[df.price<1000]
df2.groupby(['neighbourhood_group', 'neighbourhood', 'room_type'])['price'].mean().nlargest(10).reset_index()

df2= df[df.price<1000]
df2.groupby(['neighbourhood_group', 'neighbourhood', 'room_type'])['price'].median().nlargest(10).reset_index()

df2.price.describe()
