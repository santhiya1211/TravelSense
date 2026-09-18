"""
TravelSense - Geocode Cities for Map View
---------------------------------------------
Looks up latitude/longitude for every unique city in your destination
profile using OpenStreetMap's free Nominatim service, and saves a
lookup file. This only needs to be run ONCE - after that, the map
loads instantly using the cached coordinates.

IMPORTANT: Nominatim's usage policy requires max 1 request per second.
With ~1,700-1,800 unique cities, this will take roughly 30 minutes.
That's expected - let it run in the background. It saves progress
as it goes, so if it gets interrupted, just run it again and it will
skip cities it already found.
"""

import pandas as pd
import time
import os
import logging
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter

# Quiet down geopy's verbose retry/error logging so the terminal stays readable
logging.getLogger("geopy").setLevel(logging.CRITICAL)

PROFILE_PATH = r"C:\Users\santhiya\Downloads\archive (3)\all_destinations_profile.csv"
CACHE_PATH = r"C:\Users\santhiya\Downloads\archive (3)\city_coordinates.csv"

print("Loading destination profile...")
profile = pd.read_csv(PROFILE_PATH)
unique_cities = sorted(profile['City'].dropna().unique())
print(f"Found {len(unique_cities)} unique cities to geocode.")

# Load existing cache if present, so we can resume
if os.path.exists(CACHE_PATH):
    cache_df = pd.read_csv(CACHE_PATH)
    cache = dict(zip(cache_df['City'], zip(cache_df['lat'], cache_df['lon'])))
    print(f"Resuming from existing cache: {len(cache)} cities already geocoded.")
else:
    cache = {}

geolocator = Nominatim(user_agent="travelsense_project_app", timeout=10)
geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1.1, max_retries=1, error_wait_seconds=2.0)

remaining = [c for c in unique_cities if c not in cache]
print(f"Cities left to geocode: {len(remaining)}")

for i, city in enumerate(remaining):
    try:
        location = geocode(f"{city}, India")
        if location:
            cache[city] = (location.latitude, location.longitude)
        else:
            cache[city] = (None, None)
    except Exception as e:
        print(f"  Error geocoding {city}: {e}")
        cache[city] = (None, None)

    if (i + 1) % 25 == 0 or (i + 1) == len(remaining):
        print(f"  Progress: {i + 1}/{len(remaining)} geocoded...")
        # Save progress periodically
        out_df = pd.DataFrame([(c, v[0], v[1]) for c, v in cache.items()],
                               columns=['City', 'lat', 'lon'])
        out_df.to_csv(CACHE_PATH, index=False)

# Final save
out_df = pd.DataFrame([(c, v[0], v[1]) for c, v in cache.items()],
                       columns=['City', 'lat', 'lon'])
out_df.to_csv(CACHE_PATH, index=False)

found = out_df['lat'].notna().sum()
print(f"\nDONE. Geocoded {found} / {len(unique_cities)} cities successfully.")
print(f"Saved to: {CACHE_PATH}")