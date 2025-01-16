import rioxarray
import xarray as xr
import pandas as pd
from common import convert_coords
from typing import Optional, Union

def read_EnMAP(
    filepath: str,
    wavelengths: Optional[Union[list, tuple]] = None,
    method: Optional[str] = "nearest",
    **kwargs,
) -> xr.Dataset:
    """
    Reads EnMAP data from a given file and returns an xarray Dataset with proper handling of no-data values.

    Args:
        filepath (str): Path to the file to read.
        wavelengths (array-like, optional): Specific wavelengths to select. If
            None, all wavelengths are selected.
        method (str, optional): Method to use for selection when wavelengths is not
            None. Defaults to "nearest".
        **kwargs: Additional keyword arguments to pass to the `sel` method when
            bands is not None.

    Returns:
        xr.Dataset: An xarray Dataset containing the EnMAP data.
    """

    file_path_csv = r"D:\PhD_Thesis\testoo\enmap_wavelengths_full.csv"
    df = pd.read_csv(file_path_csv)

    # Open the dataset with scaling and masking enabled
    dataset = xr.open_dataset(filepath)

    # Rename dimensions and variables for consistency
    dataset = dataset.rename(
        {"band": "wavelength", "band_data": "reflectance"}
    ).transpose("y", "x", "wavelength")

    # Assign actual wavelengths from the CSV file
    dataset["wavelength"] = df["wavelength"].tolist()

    # Mask invalid data (e.g., NaN or nodata values)
    nodata_value = -32768
    dataset = dataset.where(dataset != nodata_value)

    # Normalize reflectance values if they are scaled (e.g., 0-10000)
    if dataset["reflectance"].max() > 1.0:
        dataset["reflectance"] = dataset["reflectance"] / 10000

    # Select specific wavelengths if provided
    if wavelengths is not None:
        dataset = dataset.sel(wavelength=wavelengths, method=method, **kwargs)

    # Set CRS information
    dataset.attrs["crs"] = dataset.rio.crs.to_string()

    return dataset

def EnMAP_to_image(
    dataset: Union[xr.Dataset, str],
    wavelengths: Union[list, tuple] = None,
    method: Optional[str] = "nearest",
    output: Optional[str] = None,
    **kwargs,
):
    """
    Converts an EnMAP dataset to an image with proper handling of NaN values.

    Args:
        dataset (xarray.Dataset or str): The dataset containing the EnMAP data
            or the file path to the dataset.
        wavelengths (array-like, optional): The specific wavelengths to select.
            If None, all wavelengths are selected. Defaults to None.
        method (str, optional): The method to use for data interpolation.
            Defaults to "nearest".
        output (str, optional): The file path where the image will be saved. If
            None, the image will be returned as a PIL Image object. Defaults to None.
        **kwargs: Additional keyword arguments to be passed to
            `leafmap.array_to_image`.

    Returns:
        rasterio.Dataset or None: The image converted from the dataset. If
            `output` is provided, the image will be saved to the specified file
            and the function will return None.
    """
    from leafmap import array_to_image

    if isinstance(dataset, str):
        dataset = read_EnMAP(dataset, method=method)

    if wavelengths is not None:
        dataset = dataset.sel(wavelength=wavelengths, method=method)

    # Handle NaN values by setting them to 0 (or a colormap-ignored value)
    dataset["reflectance"] = dataset["reflectance"].fillna(0)

    return array_to_image(
        dataset["reflectance"], output=output, transpose=False, **kwargs
    )

def extract_EnMAP(ds: xr.Dataset, lat: float, lon: float) -> xr.DataArray:
    """
    Extracts EnMAP data from a given xarray Dataset at a specific location.

    Args:
        ds (xarray.Dataset): The dataset containing the EnMAP data.
        lat (float): The latitude of the point to extract.
        lon (float): The longitude of the point to extract.

    Returns:
        xarray.DataArray: The extracted data.
    """

    crs = ds.attrs["crs"]

    # Convert geographic coordinates to the dataset CRS
    x, y = convert_coords([[lat, lon]], "epsg:4326", crs)[0]

    # Extract the nearest pixel's reflectance values
    values = ds.sel(x=x, y=y, method="nearest")["reflectance"].values

    return xr.DataArray(
        values, dims=["wavelength"], coords={"wavelength": ds.coords["wavelength"]}
    )

def filter_EnMAP(
    dataset: xr.Dataset,
    lat: Union[float, tuple],
    lon: Union[float, tuple],
    return_plot: Optional[bool] = False,
    **kwargs,
) -> xr.Dataset:
    """
    Filters an EnMAP dataset based on latitude and longitude.

    Args:
        dataset (xr.Dataset): The EnMAP dataset to filter.
        lat (float or tuple): The latitude to filter by. If a tuple or list,
            it represents a range.
        lon (float or tuple): The longitude to filter by. If a tuple or
            list, it represents a range.

    Returns:
        xr.DataArray: The filtered EnMAP data.
    """

    if isinstance(lat, (list, tuple)):
        min_lat, max_lat = min(lat), max(lat)
    else:
        min_lat, max_lat = lat, lat

    if isinstance(lon, (list, tuple)):
        min_lon, max_lon = min(lon), max(lon)
    else:
        min_lon, max_lon = lon, lon

    # Convert lat/lon to dataset CRS
    coords = convert_coords(
        [[min_lat, min_lon], [max_lat, max_lon]], "epsg:4326", dataset.rio.crs.to_string()
    )

    if len(coords) == 1:
        x, y = coords[0]
        filtered = dataset.sel(x=x, y=y, method="nearest")["reflectance"]
    else:
        x_min, y_min = coords[0]
        x_max, y_max = coords[1]
        filtered = dataset.sel(x=slice(x_min, x_max), y=slice(y_min, y_max))["reflectance"]

    if return_plot:
        filtered.stack({"pixel": ("y", "x")}).plot.line(hue="pixel", **kwargs)
    else:
        return filtered