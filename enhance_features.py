
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from math import radians, cos, sin, asin, sqrt
import osmnx as ox
import warnings

warnings.filterwarnings('ignore')

def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    r = 6371 # Radius of earth in kilometers
    return c * r

def calculate_diversity_vectorized(df, radius_m=2000):
    print("Fetching all amenities for Addis Ababa...")
    # Define a bounding box for Addis Ababa roughly
    # Addis is roughly between 8.8 and 9.1 Lat, 38.6 and 38.9 Lon
    try:
        # Use a slightly larger box to be safe
        tags = {'amenity': True}
        all_pois = ox.features_from_bbox(bbox=(38.6, 8.8, 38.9, 9.1), tags=tags)
        
        if all_pois.empty or 'amenity' not in all_pois.columns:
            df['amenity_diversity_2km'] = 0
            return df
            
        print(f"Found {len(all_pois)} total amenities. Calculating diversity per point...")
        
        import geopandas as gpd
        from shapely.geometry import Point
        
        # Convert POIs to a GeoDataFrame
        all_pois = all_pois[all_pois['amenity'].notna()]
        
        # Create GeoDataFrame for our data points
        gdf_points = gpd.GeoDataFrame(
            df, 
            geometry=[Point(xy) for xy in zip(df.lon, df.lat)],
            crs="EPSG:4326"
        )
        
        # Project to a metric CRS for accurate distance (UTM 37N is good for Addis)
        gdf_points = gdf_points.to_crs(epsg=32637)
        all_pois = all_pois.to_crs(epsg=32637)
        
        # For each point, find POIs within radius
        diversity_scores = []
        for i, point in gdf_points.iterrows():
            # Filter POIs within radius
            nearby_pois = all_pois[all_pois.distance(point.geometry) <= radius_m]
            diversity_scores.append(nearby_pois['amenity'].nunique())
            
        df['amenity_diversity_2km'] = diversity_scores
        return df
    except Exception as e:
        print(f"Error in vectorized diversity: {e}")
        df['amenity_diversity_2km'] = 0
        return df

def enhance_data(file_path):
    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path)
    
    # 1. Distances to Key Landmarks
    meskel_square = (38.7525, 9.0192)
    bole_airport = (38.7993, 8.9806)
    
    print("Calculating distances to landmarks...")
    df['dist_to_meskel_km'] = df.apply(lambda row: haversine(row['lon'], row['lat'], meskel_square[0], meskel_square[1]), axis=1)
    df['dist_to_bole_km'] = df.apply(lambda row: haversine(row['lon'], row['lat'], bole_airport[0], bole_airport[1]), axis=1)
    
    # 2. K-Means Clustering for Neighborhoods
    print("Performing K-Means clustering...")
    coords = df[['lat', 'lon']]
    kmeans = KMeans(n_clusters=15, random_state=42)
    df['neighborhood_cluster'] = kmeans.fit_predict(coords)
    
    # 3. Amenity Diversity Index
    df = calculate_diversity_vectorized(df)
    
    output_path = file_path.replace('.csv', '_enhanced.csv')
    df.to_csv(output_path, index=False)
    print(f"Enhanced data saved to {output_path}")
    return output_path

if __name__ == "__main__":
    enhance_data('data/addis_house_price.csv')
