import pandas as pd
from tqdm import tqdm

# Load your CSV files
weather_df = pd.read_csv(r"C:\Users\LENOVO\Desktop\semproject\merged_weather_data.csv", chunksize=50000)

def fill_jan_1_with_jan_2(chunk):
    """Create January 1st rows with values from January 2nd."""
    print("Creating January 1st rows with January 2nd values...")

    # Ensure 'date' column is in datetime format
    chunk['date'] = pd.to_datetime(chunk['date'], errors='coerce')

    # Loop through each location and year
    locations = chunk['Location'].unique()
    years = chunk['date'].dt.year.unique()

    for loc in tqdm(locations, desc="Processing Locations"):
        for year in years:
            jan_1_date = pd.Timestamp(f"{year}-01-01")
            jan_2_date = pd.Timestamp(f"{year}-01-02")

            # Check if January 2nd data exists for this location and year
            jan_2_data = chunk[(chunk['Location'] == loc) & (chunk['date'] == jan_2_date)]

            if not jan_2_data.empty:
                # Create a new row for January 1st by copying January 2nd data
                jan_2_values = jan_2_data.iloc[0]
                jan_1_row = jan_2_values.copy()
                jan_1_row['date'] = jan_1_date  # Set the date to January 1st
                jan_1_row['Location'] = loc  # Ensure the location is correct

                # Append this row to the DataFrame
                chunk = pd.concat([chunk, pd.DataFrame([jan_1_row])], ignore_index=True)

    # Sort by Location and Date to maintain proper order
    chunk = chunk.sort_values(by=['Location', 'date']).reset_index(drop=True)

    return chunk

def preprocess_weather_data(weather_df):
    """Preprocess weather data."""
    print("Processing weather data in chunks...")

    weather_df_preprocessed = pd.DataFrame()

    for chunk_idx, chunk in enumerate(tqdm(weather_df, desc="Reading Chunks")):
        print(f"Processing chunk {chunk_idx + 1}...")
        
        # Create January 1st rows with January 2nd values
        chunk = fill_jan_1_with_jan_2(chunk)
        
        # Concatenate processed chunk to the final DataFrame
        weather_df_preprocessed = pd.concat([weather_df_preprocessed, chunk], ignore_index=True)

    return weather_df_preprocessed

# Apply preprocessing
weather_df_preprocessed = preprocess_weather_data(weather_df)

# Save the preprocessed data
weather_df_preprocessed.to_csv(r"C:\Users\LENOVO\Desktop\semproject\preprocessed_weather_data.csv", index=False)

print("Preprocessing completed and data saved.")
