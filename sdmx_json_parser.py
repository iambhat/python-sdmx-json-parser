import pandas as pd
import json

def parse_sdmx_json_to_dataframe(json_data):
    """
    Parse SDMX-JSON format data into a pandas DataFrame.
    
    Parameters:
    -----------
    json_data : dict or str
        SDMX-JSON data as dictionary or JSON string
    
    Returns:
    --------
    pd.DataFrame
        Parsed data with dimensions as columns and observations as rows
    """
    
    # If json_data is a string, parse it
    if isinstance(json_data, str):
        json_data = json.loads(json_data)
    
    # Extract structure and data
    structure = json_data['data']['structures'][0]
    dataset = json_data['data']['dataSets'][0]
    
    # Get dimension definitions
    series_dims = structure['dimensions']['series']
    obs_dims = structure['dimensions']['observation']
    
    # Create dimension lookup dictionaries
    dim_lookups = {}
    dim_names = []
    
    # Process series dimensions
    for dim in series_dims:
        dim_id = dim['id']
        dim_names.append(dim_id)
        dim_lookups[dim_id] = {i: val['id'] for i, val in enumerate(dim['values'])}
    
    # Process observation dimensions (usually TIME_PERIOD)
    for dim in obs_dims:
        dim_id = dim['id']
        dim_names.append(dim_id)
        dim_lookups[dim_id] = {i: val.get('value', val.get('id')) for i, val in enumerate(dim['values'])}
    
    # Parse series data
    rows = []
    series_data = dataset['series']
    
    for series_key, series_value in series_data.items():
        # Parse series key (e.g., "0:9:0" -> [0, 9, 0])
        series_indices = [int(x) for x in series_key.split(':')]
        
        # Get dimension values for this series
        series_dims_values = {}
        for i, dim_name in enumerate(dim_names[:-1]):  # Exclude observation dimension
            dim_value = dim_lookups[dim_name][series_indices[i]]
            series_dims_values[dim_name] = dim_value
        
        # Parse observations
        observations = series_value.get('observations', {})
        
        for obs_key, obs_value in observations.items():
            # Get time period or observation dimension value
            obs_dim_name = dim_names[-1]
            obs_index = int(obs_key)
            obs_dim_value = dim_lookups[obs_dim_name][obs_index]
            
            # Create row
            row = series_dims_values.copy()
            row[obs_dim_name] = obs_dim_value
            
            # Add observation value (first element in the array)
            row['OBS_VALUE'] = obs_value[0]
            
            # Optionally add attributes (second and third elements)
            if len(obs_value) > 1 and obs_value[1] is not None:
                row['PRECISION'] = obs_value[1]
            if len(obs_value) > 2:
                row['DERIVATION_TYPE'] = obs_value[2]
            
            rows.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(rows)
    
    # Convert OBS_VALUE to numeric
    df['OBS_VALUE'] = pd.to_numeric(df['OBS_VALUE'], errors='coerce')
    
    # Sort by dimensions for better readability
    sort_cols = [col for col in dim_names if col in df.columns]
    df = df.sort_values(by=sort_cols).reset_index(drop=True)
    
    return df


# Example usage
if __name__ == "__main__":
    # Load your JSON data
    # Option 1: From file
    with open('data.json', 'r') as f:
        json_data = json.load(f)
    
    # Option 2: From JSON string
    # json_string = '''your_json_string_here'''
    
    # Parse to DataFrame
    df = parse_sdmx_json_to_dataframe(json_data)
    
    # Display results
    print(df.head())
    print(f"\nDataFrame shape: {df.shape}")
    print(f"\nColumns: {df.columns.tolist()}")
    
    # Optional: Pivot for analysis
    # df_pivot = df.pivot_table(
    #     values='OBS_VALUE',
    #     index=['COUNTRY', 'TIME_PERIOD'],
    #     columns='INDICATOR'
    # )
    # print(df_pivot.head())
    

