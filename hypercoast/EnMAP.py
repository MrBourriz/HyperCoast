import rioxarray
import xarray as xr
import pandas as pd
from common import convert_coords
from typing import Optional, Union

def read_enmap(
    filepath: str,
    bands: Optional[Union[list, tuple]] = None,
    method: Optional[str] = "nearest",
    metadata_url: Optional[str] = None,
    nodata_value: Optional[float] = -32768,
    fill_value: Optional[float] = 0,
    **kwargs,
) -> xr.Dataset:
     """
    Reads a generic dataset and returns an xarray Dataset.

    Args:
        filepath (str): Path to the file to read.
        bands (array-like, optional): Specific bands or wavelengths to select.
            If None, all bands are selected.
        method (str, optional): Selection method for bands. Defaults to "nearest".
        metadata_url (str, optional): URL to additional metadata (e.g., wavelength mappings).
        **kwargs: Additional arguments for selecting data.

    Returns:
        xr.Dataset: The dataset.
    """
     if metadata_url:
        metadata = pd.read_csv(metadata_url)  # Example for CSV-based metadata
        # Adjust metadata usage as per the new dataset
     else:
        metadata = None

     dataset = xr.open_dataset(filepath,mask_and_scale=True)
    # Rename or transpose dimensions if necessary
     if "band" in dataset.dims:
        dataset = dataset.rename({"band": "wavelength",'band_data':'reflectance'}).transpose("y", "x", "wavelength")

     if metadata is not None and "wavelength" in metadata.columns:
        dataset["wavelength"] = metadata["wavelength"].tolist()

     if bands is not None:
        dataset = dataset.sel(wavelength=bands, method=method, **kwargs)

     dataset.attrs["crs"] = dataset.rio.crs.to_string() if dataset.rio.crs else None
     # Replace nodata and NaN values with the fill value
     if nodata_value is not None:
        dataset["reflectance"] = dataset["reflectance"].where(
            dataset["reflectance"] != nodata_value
        )
     dataset["reflectance"] = dataset["reflectance"].fillna(fill_value)

     return dataset




def enmap_to_image(
    dataset: Union[xr.Dataset, str],
    wavelengths: Union[list, tuple] = None,
    variable: str = "reflectance",
    output: Optional[str] = None,
    **kwargs,
):
    """
    Converts EnMAP data to an image.

    Args:
        dataset (xarray.Dataset or str): Dataset or file path.
        wavelengths (list or tuple, optional): Wavelengths to select.
        variable (str): Variable to convert to an image.
        output (str, optional): File path to save the image.

    Returns:
        Image or None: The generated image or None if saved.
    """
    from leafmap import array_to_image

    if isinstance(dataset, str):
        dataset = read_enmap(dataset)

    if wavelengths is not None:
        dataset = dataset.sel(wavelength=wavelengths, method="nearest")

    return array_to_image(dataset["reflectance"], output=output, **kwargs)


def extract_enmap(
    ds: xr.Dataset, lat: float, lon: float, variable: str = "reflectance"
) -> xr.DataArray:
    """
    Extracts data for a specific latitude and longitude from the EnMAP dataset.

    Args:
        ds (xarray.Dataset): The dataset.
        lat (float): Latitude.
        lon (float): Longitude.
        variable (str): The variable to extract (default: "reflectance").

    Returns:
        xarray.DataArray: Extracted data for the specified location.
    """
    crs = ds.attrs.get("crs", "EPSG:4326")
    x, y = convert_coords([[lat, lon]], "epsg:4326", crs)[0]

    values = ds.sel(x=x, y=y, method="nearest")[variable].values

    da= xr.DataArray(
        values, dims=["wavelength"], coords={"wavelength": ds.coords["wavelength"]}
    )
    return da


def filter_enmap(
    dataset: xr.Dataset,
    lat: Union[float, tuple],
    lon: Union[float, tuple],
    variable: str = "reflectance",
    return_plot: Optional[bool] = False,
    **kwargs,
) -> Union[xr.DataArray, None]:
    """
    Filters an EnMAP dataset for a specific spatial region.

    Args:
        dataset (xr.Dataset): EnMAP dataset.
        lat (float or tuple): Latitude range or single point.
        lon (float or tuple): Longitude range or single point.
        variable (str): Variable to filter (default: "reflectance").
        return_plot (bool): Whether to return a plot.
        **kwargs: Additional keyword arguments for plotting.

    Returns:
        Union[xr.DataArray, None]: Filtered data or None if plotting.
    """
    # Ensure lat and lon are tuples for ranges
    if isinstance(lat, (list, tuple)):
        min_lat, max_lat = min(lat), max(lat)
    else:
        min_lat, max_lat = lat, lat

    if isinstance(lon, (list, tuple)):
        min_lon, max_lon = min(lon), max(lon)
    else:
        min_lon, max_lon = lon, lon

    # Convert coordinates to the dataset's CRS
    try:
        coords = [[min_lat, min_lon], [max_lat, max_lon]]
        coords = convert_coords(coords, "epsg:4326", dataset.rio.crs.to_string())
    except Exception as e:
        raise ValueError(f"Coordinate conversion failed: {e}")

    # Select data based on coordinates
    if len(coords) == 1:  # Single point selection
        x, y = coords[0]
        data = dataset.sel(x=x, y=y, method="nearest")[variable]
    else:  # Range selection
        x_min, y_min = coords[0]
        x_max, y_max = coords[1]
        data = dataset.sel(x=slice(x_min, x_max), y=slice(y_min, y_max))[variable]

    # Check if the filtered data is empty
    if data.size == 0:
        raise ValueError("No data found for the specified coordinates or range.")

    # Plot the data if required
    if return_plot:
        if len(data.dims) == 1:  # 1D data, e.g., wavelength
            data.plot.line(**kwargs)
        elif len(data.dims) == 2:  # 2D data, e.g., spatial
            data.plot.imshow(**kwargs)
        else:
            raise ValueError("Data dimensionality not supported for plotting.")
    else:
        return data