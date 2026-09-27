import pandas as pd
import geopandas as gpd
import osmnx as ox
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

gdf = gpd.read_file('../data/aa_worda_boundary.zip')

print("Shape of data:", gdf.shape)
print("\nColumn names and types:")
print(gdf.dtypes)
print("\nGemoetry type:")
print(gdf.geometry.type.unique())
print("\nCRS (Coordinate Reference System):")
print(gdf.crs)
print("\nBasic Statistics:")
print(gdf.describe())

pd.set_option('display.max_rows', 150)

gdf = gdf.to_crs('EPSG:4326') 
gdf['centroid'] = gdf.geometry.centroid
gdf['lon'] = gdf['centroid'].x
gdf['lat'] = gdf['centroid'].y

gdf.drop(columns=['FID_1', 'OBJECTID', 'Shape_Le_1',	'Shape_Area', 'Region'], inplace=True)
gdf.rename(columns={'Sub_City': 'subcity'}, inplace=True)
gdf.rename(columns={'Woreda': 'district'}, inplace=True)

tags = {'amenity': ['school', 'hospital', 'bank', 'restaurant', 'cafe', 'pharmacy', 'police', 'fuel', 'parking', 'university', 'library', 'atm']}

radii = [500, 1000, 2000, 3000, 4000, 5000]

for radius in radii:
    radius_km = radius/1000
    gdf[f'amenities_within_{radius_km}km'] = 0

for index, row in gdf.iterrows():
    lat = row['lat']
    lon = row['lon']
    
    radius_counts = []
    
    for radius in radii:
        try:
            pois = ox.features_from_point((lat, lon), tags=tags, dist=radius)
            count = len(pois)
        except Exception as e:
            count = 0
        
        radius_counts.append(count)
    for i, radius in enumerate(radii):
        radius_km = radius/1000
        gdf.at[index, f'amenities_within_{radius_km}km'] = radius_counts[i]
    
    print(f"Subcity: {row['subcity']}, District: {row['district']} - "
          f"Counts: {', '.join([f'{r/1000}km:{c}' for r, c in zip(radii, radius_counts)])}")

road_types = ['primary', 'secondary', 'tertiary', 'unclassified', 'residential']
tags = {'highway': road_types}

for index, row in gdf.iterrows():
    lat = row['lat']
    lon = row['lon']
    
    radius_counts = []
    
    for radius in radii:
        try:
            # Ensure both lines below have the SAME indentation (4 spaces)
            roads = ox.features_from_point((lat, lon), tags=tags, dist=radius)
            count = len(roads)
        except Exception as e:
            print(f"Error at radius {radius}: {e}")
            count = 0
        
        radius_counts.append(count)
    
    for i, radius in enumerate(radii):
        radius_km = radius / 1000
        gdf.at[index, f'roads_within_{radius_km}km'] = radius_counts[i]
    
    counts_str = ', '.join([f'{r/1000}km:{c}' for r, c in zip(radii, radius_counts)])
    print(f"Subcity: {row['subcity']}, District: {row['district']} - "
          f"Road Counts: {counts_str}")

gdf.to_csv('../data/final_feature.csv', index=False)
