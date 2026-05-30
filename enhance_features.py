
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

import requests
import time

def fetch_elevation_batch(latitudes, longitudes):
    """
    Fetch elevation for a batch of coordinates using Open-Meteo API with retries
    """
    url = "https://elevation-api.open-meteo.com/v1/elevation"
    params = {
        "latitude": ",".join(map(str, latitudes)),
        "longitude": ",".join(map(str, longitudes))
    }
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = requests.get(url, params=params, timeout=30)
            if response.status_code == 200:
                return response.json().get('elevation', [])
            else:
                print(f"API Error (Attempt {attempt+1}): {response.status_code}")
        except Exception as e:
            print(f"Request failed (Attempt {attempt+1}): {e}")
        time.sleep(2) # Wait before retry
    
    return [0] * len(latitudes)

def add_topography_features(df):
    print("Fetching Elevation and calculating Slope (batch mode)...")
    # For each point, we'll fetch its elevation AND the elevation of 4 neighbors
    # to estimate the slope. Offset of ~30m is approx 0.00027 degrees.
    offset = 0.00027
    
    # Use unique locations to save API calls
    unique_locs = df[['lat', 'lon']].drop_duplicates().copy()
    
    all_lats = []
    all_lons = []
    
    for _, row in unique_locs.iterrows():
        lat, lon = row['lat'], row['lon']
        # Point, North, South, East, West
        all_lats.extend([lat, lat + offset, lat - offset, lat, lat])
        all_lons.extend([lon, lon, lon, lon + offset, lon - offset])
    
    # Fetch in chunks of 50 properties (250 coordinates) to avoid URL length limits
    batch_size = 250
    elevations = []
    print(f"Total coordinates to fetch: {len(all_lats)}")
    
    for i in range(0, len(all_lats), batch_size):
        chunk_lats = all_lats[i:i + batch_size]
        chunk_lons = all_lons[i:i + batch_size]
        elevations.extend(fetch_elevation_batch(chunk_lats, chunk_lons))
        print(f"Fetched {min(i + batch_size, len(all_lats))}/{len(all_lats)} elevations...")
        time.sleep(0.1) # Respect API
        
    # Process elevations back into slope
    results = []
    for i in range(0, len(elevations), 5):
        try:
            p_elev = elevations[i]
            n_elev = elevations[i+1]
            s_elev = elevations[i+2]
            e_elev = elevations[i+3]
            w_elev = elevations[i+4]
            
            # Max difference in elevation over ~30m distance
            max_diff = max(abs(n_elev - s_elev), abs(e_elev - w_elev)) / 60.0 # roughly 60m between N-S or E-W
            slope = np.arctan(max_diff) * (180/np.pi) # Degrees
            
            results.append({'lat': all_lats[i], 'lon': all_lons[i], 'elevation': p_elev, 'slope': slope})
        except:
            results.append({'lat': all_lats[i], 'lon': all_lons[i], 'elevation': 0, 'slope': 0})
            
    topo_df = pd.DataFrame(results)
    df = df.merge(topo_df, on=['lat', 'lon'], how='left')
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
    
    # 4. Topography Features (Elevation and Slope)
    df = add_topography_features(df)
    
    output_path = file_path.replace('.csv', '_enhanced.csv')
    df.to_csv(output_path, index=False)
    print(f"Enhanced data saved to {output_path}")
    return output_path

if __name__ == "__main__":
    enhance_data('data/addis_house_price.csv')
