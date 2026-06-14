#!/usr/bin/env python3
"""Convert Last.fm CSV from text format to numeric format for ML algorithms."""
import csv
from collections import defaultdict
import numpy as np

# Load the original CSV
data = []
users_set = set()
artists_set = set()

with open('Last.fm_data.csv', 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        username = row['Username'].strip()
        artist = row['Artist'].strip()
        data.append((username, artist))
        users_set.add(username)
        artists_set.add(artist)

print(f"Original data points: {len(data)}")
print(f"Unique users: {len(users_set)}")
print(f"Unique artists: {len(artists_set)}")

# Create interaction matrix (play counts)
interactions = defaultdict(lambda: defaultdict(int))
for username, artist in data:
    interactions[username][artist] += 1

# Create numeric mappings
users = sorted(list(users_set))
artists = sorted(list(artists_set))

user_to_id = {u: i for i, u in enumerate(users)}
artist_to_id = {a: i for i, a in enumerate(artists)}

print(f"\nCreated {len(user_to_id)} user IDs and {len(artist_to_id)} artist IDs")
print(f"\nSample mapping:")
print(f"  User: {users[0]} -> ID {user_to_id[users[0]]}")
print(f"  Artist: {artists[0]} -> ID {artist_to_id[artists[0]]}")

# Create numeric CSV with all interactions
numeric_data = []
for username in interactions:
    for artist in interactions[username]:
        user_id = user_to_id[username]
        artist_id = artist_to_id[artist]
        plays = interactions[username][artist]
        numeric_data.append([user_id, artist_id, plays])

# Save as CSV
with open('Last.fm_data_numeric.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['user_id', 'artist_id', 'plays'])  # header
    writer.writerows(numeric_data)

print(f"\nSaved numeric CSV with {len(numeric_data)} interactions")
print("First 10 rows:")
for row in numeric_data[:10]:
    print(f"  {row}")

# Verify it can be loaded
arr = np.array(numeric_data)
X, y = arr[:, :-1], arr[:, -1]
print(f"\nVerified load: X shape {X.shape}, y shape {y.shape}")
print("\n✓ CSV format conversion complete!")
