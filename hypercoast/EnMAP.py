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
    Reads EnMAP data from a given file and returns an xarray Dataset.

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

    file_path_csv = r"D:\PhD_Thesis\Contributions\enmap_wavelengths_full.csv"
    df = pd.read_csv(file_path_csv)
    dataset = xr.open_dataset(filepath)
    dataset = dataset.rename(
        {"band": "wavelength", "band_data": "reflectance"}
    ).transpose("y", "x", "wavelength")
    dataset["wavelength"] = df["wavelength"].tolist()

    if wavelengths is not None:
        dataset = dataset.sel(wavelength=wavelengths, method=method, **kwargs)

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
    Converts an EnMAP dataset to an image.

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

    return array_to_image(
        dataset["reflectance"], output=output, transpose=False, **kwargs
    )

path_enmap=r'D:\PhD_Thesis\Contributions\EnMAP.tif'

dataset_enmap=read_EnMAP(path_enmap)
print(dataset_enmap)