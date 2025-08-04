import pandas as pd

# Load the merged dataset
merged_df = pd.read_csv("merged_weather_locations_cleaned.csv")

# Drop unnecessary columns: 'Location_cleaned_x', 'Location_aligned', 'Location_y', 'Location_cleaned_y', and 'Altitude'
merged_df = merged_df.drop(columns=['.geo'])

# Save the updated dataset back to a CSV file
merged_df.to_csv("merged_weather_locations_cleaned_2.csv", index=False)
