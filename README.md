# MDBA-Inundation-History-Tool

## Tool purpose
This tool is used for associating river flow volumes with Landsat satellite passes. It filters satellite passes within defined flow bands of interest, removes poor quality satellite data, and also applies a filter to those images on the rising/falling limb. The images are then analysed using the Fisher index and NDWI for water identification in the landscape. 
The tool outputs the images as NetCDF files for use in GIS software.

## Compatible software environment
This tool has been designed to run in the DEA Sandbox environment and relies on the Datacube. For more information and how to gain access, please visit: https://docs.dea.ga.gov.au/setup/sandbox.html
The sandbox is free to use.

## Support
For **issues relating to the script, a tutorial, or feedback** please contact Martin Job at martin.job@mdba.gov.au or David Weldrake at david.weldrake@mdba.gov.au

## Setup
1. Establish access to the Sandbox and clone this repository into your local environment.
2. Create three folders in the "Main" folder for storing the program outputs, titled:
    - "csv_outputs"
    - "hydrograph_outputs"
    - "netcdf_outputs"
3. Open the "Inundation_History_Tool.ipynb" file 
4. Run the script, using the notes within the notebook to guide you.