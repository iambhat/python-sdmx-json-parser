"""
SDMX-JSON Data Parser
=====================
Professional module for parsing SDMX-JSON format data from IMF API into pandas DataFrames.

Author: Data Analytics Team
Date: 2025
Version: 1.0
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError

import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SDMXParserError(Exception):
    """Custom exception for SDMX parsing errors."""
    pass


class SDMXDataFetcher:
    """
    Handles fetching data from SDMX API endpoints.
    """
    
    DEFAULT_HEADERS = {
        'Cache-Control': 'no-cache',
        'Accept': 'application/json',
        'User-Agent': 'Python-SDMX-Client/1.0'
    }
    
    def __init__(self, base_url: str, timeout: int = 30):
        """
        Initialize the data fetcher.
        
        Args:
            base_url: Base URL for the SDMX API
            timeout: Request timeout in seconds
        """
        self.base_url = base_url
        self.timeout = timeout
    
    def fetch_data(self, endpoint: str = "", headers: Optional[Dict] = None) -> Dict:
        """
        Fetch data from the API endpoint.
        
        Args:
            endpoint: API endpoint path (appended to base_url)
            headers: Optional custom headers
            
        Returns:
            Parsed JSON response as dictionary
            
        Raises:
            SDMXParserError: If request fails or response is invalid
        """
        url = f"{self.base_url}/{endpoint}".rstrip('/')
        request_headers = {**self.DEFAULT_HEADERS, **(headers or {})}
        
        try:
            logger.info(f"Fetching data from: {url}")
            request = Request(url, headers=request_headers)
            
            with urlopen(request, timeout=self.timeout) as response:
                if response.status != 200:
                    raise SDMXParserError(f"HTTP {response.status}: {response.reason}")
                
                data = response.read().decode('utf-8')
                return json.loads(data)
                
        except HTTPError as e:
            raise SDMXParserError(f"HTTP Error {e.code}: {e.reason}") from e
        except URLError as e:
            raise SDMXParserError(f"URL Error: {e.reason}") from e
        except json.JSONDecodeError as e:
            raise SDMXParserError(f"Invalid JSON response: {e}") from e
        except Exception as e:
            raise SDMXParserError(f"Unexpected error: {e}") from e


class SDMXParser:
    """
    Parses SDMX-JSON format data into pandas DataFrames.
    """
    
    @staticmethod
    def _validate_structure(json_data: Dict) -> None:
        """
        Validate the SDMX-JSON structure.
        
        Args:
            json_data: The JSON data to validate
            
        Raises:
            SDMXParserError: If structure is invalid
        """
        try:
            if 'data' not in json_data:
                raise SDMXParserError("Missing 'data' key in JSON")
            
            data = json_data['data']
            if 'structures' not in data or not data['structures']:
                raise SDMXParserError("Missing or empty 'structures' in data")
            
            if 'dataSets' not in data or not data['dataSets']:
                raise SDMXParserError("Missing or empty 'dataSets' in data")
                
        except (KeyError, TypeError, IndexError) as e:
            raise SDMXParserError(f"Invalid SDMX-JSON structure: {e}") from e
    
    @staticmethod
    def _build_dimension_lookups(structure: Dict) -> tuple[Dict[str, Dict], List[str]]:
        """
        Build dimension lookup dictionaries.
        
        Args:
            structure: Structure definition from SDMX-JSON
            
        Returns:
            Tuple of (dimension_lookups, dimension_names)
        """
        dim_lookups = {}
        dim_names = []
        
        # Process series dimensions
        for dim in structure['dimensions']['series']:
            dim_id = dim['id']
            dim_names.append(dim_id)
            dim_lookups[dim_id] = {
                idx: val['id'] 
                for idx, val in enumerate(dim['values'])
            }
        
        # Process observation dimensions
        for dim in structure['dimensions']['observation']:
            dim_id = dim['id']
            dim_names.append(dim_id)
            dim_lookups[dim_id] = {
                idx: val.get('value', val.get('id')) 
                for idx, val in enumerate(dim['values'])
            }
        
        return dim_lookups, dim_names
    
    @staticmethod
    def _parse_series_data(
        series_data: Dict,
        dim_lookups: Dict[str, Dict],
        dim_names: List[str]
    ) -> List[Dict]:
        """
        Parse series data into list of row dictionaries.
        
        Args:
            series_data: Series data from dataset
            dim_lookups: Dimension lookup dictionaries
            dim_names: List of dimension names
            
        Returns:
            List of row dictionaries
        """
        rows = []
        obs_dim_name = dim_names[-1]  # Last dimension is observation dimension
        
        for series_key, series_value in series_data.items():
            # Parse series indices
            series_indices = list(map(int, series_key.split(':')))
            
            # Build base row with series dimensions
            base_row = {
                dim_name: dim_lookups[dim_name][series_indices[i]]
                for i, dim_name in enumerate(dim_names[:-1])
            }
            
            # Process observations
            observations = series_value.get('observations', {})
            
            for obs_key, obs_value in observations.items():
                obs_index = int(obs_key)
                
                # Create row
                row = base_row.copy()
                row[obs_dim_name] = dim_lookups[obs_dim_name][obs_index]
                row['OBS_VALUE'] = obs_value[0]
                
                # Add optional attributes
                if len(obs_value) > 1 and obs_value[1] is not None:
                    row['PRECISION'] = obs_value[1]
                if len(obs_value) > 2:
                    row['DERIVATION_TYPE'] = obs_value[2]
                
                rows.append(row)
        
        return rows
    
    def parse_to_dataframe(
        self, 
        json_data: Union[Dict, str],
        sort_output: bool = True
    ) -> pd.DataFrame:
        """
        Parse SDMX-JSON data into a pandas DataFrame.
        
        Args:
            json_data: SDMX-JSON data as dictionary or JSON string
            sort_output: Whether to sort the output by dimensions
            
        Returns:
            Parsed DataFrame with dimensions and observations
            
        Raises:
            SDMXParserError: If parsing fails
        """
        try:
            # Parse JSON string if needed
            if isinstance(json_data, str):
                json_data = json.loads(json_data)
            
            # Validate structure
            self._validate_structure(json_data)
            
            # Extract structure and dataset
            structure = json_data['data']['structures'][0]
            dataset = json_data['data']['dataSets'][0]
            
            # Build dimension lookups
            dim_lookups, dim_names = self._build_dimension_lookups(structure)
            
            # Parse series data
            rows = self._parse_series_data(
                dataset['series'],
                dim_lookups,
                dim_names
            )
            
            if not rows:
                logger.warning("No data rows parsed from SDMX-JSON")
                return pd.DataFrame()
            
            # Create DataFrame
            df = pd.DataFrame(rows)
            
            # Convert observation value to numeric
            df['OBS_VALUE'] = pd.to_numeric(df['OBS_VALUE'], errors='coerce')
            
            # Sort by dimensions if requested
            if sort_output:
                sort_cols = [col for col in dim_names if col in df.columns]
                df = df.sort_values(by=sort_cols).reset_index(drop=True)
            
            logger.info(f"Successfully parsed {len(df)} observations")
            return df
            
        except json.JSONDecodeError as e:
            raise SDMXParserError(f"Invalid JSON: {e}") from e
        except Exception as e:
            raise SDMXParserError(f"Parsing error: {e}") from e


def main():
    """
    Main execution function.
    """
    # Configuration
    API_URL = "https://api.imf.org/external/sdmx/3.0/data"
    OUTPUT_FILE = "sdmx_data_output.csv"
    
    try:
        # Initialize fetcher and parser
        # fetcher = SDMXDataFetcher(base_url=API_URL) # uncomment with actual json data url
        parser = SDMXParser()
        
        # Fetch data
        logger.info("Starting data fetch...")
        #json_data = fetcher.fetch_data()

        # test with sample json data from script folder
        with open('data.json', 'r') as f:
            json_data = json.load(f)
        
        # Parse to DataFrame
        logger.info("Parsing data to DataFrame...")
        df = parser.parse_to_dataframe(json_data)
        
        # Display summary
        logger.info(f"DataFrame shape: {df.shape}")
        logger.info(f"Columns: {df.columns.tolist()}")
        logger.info(f"\nFirst few rows:\n{df.head()}")
        
        # Save to CSV
        output_path = Path(OUTPUT_FILE)
        df.to_csv(output_path, index=False)
        logger.info(f"Data saved to: {output_path.absolute()}")
        
        return df
        
    except SDMXParserError as e:
        logger.error(f"SDMX Parser Error: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise


if __name__ == "__main__":
    main()
