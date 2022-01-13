# MDBA-Inundation-History-Tool (IHT 0.0.2)

## Support
For **issues relating to the script, a tutorial, or feedback** please contact Martin Job at martin.job@mdba.gov.au, Gabrielle Hunt at gabrielle.hunt@mdba.gov.au or Beau Byers at beau.byers@mdba.gov.au:

## Tool purpose
This tool is used for associating river flow rates with Landsat and Sentinel satellite passes which are accessed through [Google Earth Engine](https://developers.google.com/earth-engine). It filters satellite passes within defined flow bands of interest, removes cloudy imagery, and also applies a filter to those images on the rising/falling limb. The images are then indexed using the MNDWI, NDWI, Fisher index and a custom False colour (swir2, nir and red) index for water identification.

The tool outputs the images as Geotiff files to Google Cloud, Google Drive or as an Asset for use in GIS software (see [Exporting Data](https://developers.google.com/earth-engine/guides/exporting) user guide).

> **Note:** To use this notebook you will need the have a Google Earth Engine account, subject to their [Terms and Conditons](https://earthengine.google.com/terms/)

This tool has been designed to run in Jupyter Notebooks.


## Quick use notes
Prior to running the tool, ensure you have set up your environment using the requirements.txt file provided in the repo.

1. Authenticate to GEE. Right-click on the link to open in a new tab, sign in, copy the code, paste it into the box and press 'enter' only.
2. Press shift + enter on following cells until IHT Dashboard appears. Use the dashboard to input your selections. Start by selecting the Jupyter environment you are deploying the tool in, then a date range of interest, then defining the flow band of interest. You then have three options (select one) for defining the extents of your location of interest. Extents can be defined by either uploading a shapefile, typing in the coordinates manually, or using the coordinates of the gauge you selected in the previous step. You can then type in a gauge number and select if you want to include landsat 7 passes with the SLC failure (leading to stripes in these images)
3. Press shift + enter until you get to 'PART 2'. Here you can toggle the satellite passes you wish to analyse and export based on their position on the hydrograph.
5. Open your google cloud storage and download the Geotiff files for import into your preferred GIS software.
6. OPTIONAL: download any of the plots of interest.
7. OPTIONAL: Download the csv, containing the date of the satellite flyover, gauge reading, and whether you considered it a pass or fail.
8. Once you have finished, restart the kernal to remove the variables and rerun the cells with your next parameters