# +
# %matplotlib inline
import datetime
import copy
import pprint
from typing import List, Tuple
import pandas as pd
import geopandas as gpd
from ipywidgets.widgets import HBox, VBox, Text, Tab, DatePicker, Label, Dropdown, \
    BoundedFloatText, ToggleButtons, Accordion, Checkbox, BoundedIntText



# Where is the program being run:
deploy_environment = ToggleButtons(
    options=['AZML', 'Google colab', 'other'],
    description='',
    disabled=False,
    button_style='',
    tooltips=[
        'Running in Azure Machine Learning notebook',
        'Running in Google colab',
        'Running in Jupyter on another type of deployed environment'
    ],
    style={'description_width': 'initial'}
)

start_date = DatePicker(description='Start date:', disabled=False)
end_date = DatePicker(description='End date', disabled=False)

# flow and buffer
min_flow = BoundedIntText(
    value=0,
    min=0,
    max=999999,
    step=1,
    description='Minimum flow:',
    disabled=False,
    style={'description_width': 'initial'})

max_flow = BoundedIntText(
    value=0,
    min=0,
    max=999999,
    step=1,
    description='Maximum flow:',
    disabled=False,
    style={'description_width': 'initial'})

buffer_value = BoundedFloatText(
    value=0.2,
    min=0,
    max=1,
    step=0.01,
    description='Buffer around location:',
    disabled=False,
    style={'description_width': 'initial'}
)

#Gauge option
check_gauge = Checkbox(
    value=False,
    description='Use gauge location',
    disabled=False,
    indent=False,
    style={'description_width': 'initial'}
)

use_gauge_location = Accordion(children=[check_gauge])

# user option
check_own = Checkbox(
    value=False,
    description='Define my own location',
    disabled=False,
    indent=False,
    style={'description_width': 'initial'}
)
input_lat = BoundedFloatText(
    value=0.0,
    min=-99999999999999,
    max=999999999999999,
    step=0.0001,
    description='Latitude:',
    disabled=False
)

input_lon = BoundedFloatText(
    value=0.0,
    min=-99999999999999,
    max=9999999999999,
    step=0.0001,
    description='Longitude:',
    disabled=False
)

use_my_own_location = Accordion(children=[VBox([check_own, input_lat, input_lon])])

#Upload shapefile:
check_shapefile = Checkbox(
    value=False,
    description='Use shapefile from shapefile_input folder:',
    disabled=False,
    indent=False,
    style={'description_width': 'initial'}
)

shapefile_loc = Text(value='shapefile_inputs/',
                     placeholder='Enter the name of the shapefile in the folder',
                     description='Name of shapefile:',
                     disabled=False,
                     style={'description_width': 'initial'})

upload_shapefile = Accordion(children=[VBox([check_shapefile, shapefile_loc])])

# Select representative gauge:
gauge_num = Text(
    value='409025',
    placeholder='Enter gauge number',
    description='',
    disabled=False)

# What water threshold
cloud_threshold = BoundedFloatText(
    value=5,
    min=0,
    max=100,
    description='Cloud threshold:',
    disabled=False,
    style={'description_width': 'initial'})

# Landsat 7 SLC error option:
slc_option = ToggleButtons(
    options=['Exclude', 'Include'],
    description='',
    disabled=False,
    button_style='',
    tooltips=[
        'Include those Landsat 7 passes with the slc failure (stripes going through these passes)',
        'Exclude those Landsat 7 passes with the slc failure (leads to some data gaps)'
    ],
    style={'description_width': 'initial'}
)

# Which band to export
band_option = Dropdown(
    options=['MNDWI', 'Fisher', 'Cloud', 'RGB', 'False_colour'],
    value='RGB',
    description='Band:',
    disabled=False,
)

# Which band to export
summary_band_option = Dropdown(
    options=['MNDWI', 'Fisher'],
    value='MNDWI',
    description='Band:',
    disabled=False,
    style={'width': 'max-content'}
)

# What water threshold
water_threshold = BoundedFloatText(
    value=0.1,
    min=-1,
    max=1,
    description='Threshold (applied to summary band):',
    disabled=False,
    style={'description_width': 'initial'})

# Export single date images
export_images = Checkbox(
    value=False,
    description='Export to Drive',
    disabled=False,
    indent=False,
    layout={'width': 'max-content'}, # If the items' names are long
)

# Export single date images
export_summary_images = Checkbox(
    value=False,
    description='Export to Drive',
    disabled=False,
    indent=False,
    layout={'width': 'max-content'} # If the items' names are long
)


lbl = Label(value=('Select parameters for exporting single date images - '
                   'WARNING can be hundreds of images:'),
            style={'description_width': 'initial'})
single_date_export_label = HBox([lbl])

lbl = Label(value=('Select parameters for exporting summary date images - i.e. '
                   'the percentage (%) of inundation:'),
            style={'description_width': 'initial'})
summary_date_export_label = HBox([lbl])

export_variables = Accordion(children=[VBox([single_date_export_label, band_option,export_images,
                                             summary_date_export_label, summary_band_option,
                                             water_threshold, export_summary_images])])

# Labels:
use_gauge_location.set_title(0, 'Use coordinates for the gauge to define extents')
use_my_own_location.set_title(0, 'Input coordinates to define extents')
upload_shapefile.set_title(0, 'Upload shapefile to define extents')
export_variables.set_title(0, 'Export variables')


deploy_env_label = HBox([Label(value="1. Select deployment environment")])
date_selection_label = HBox([Label(value="2. Select date range of interest")])
flow_label = HBox([Label(value="3. Set lower and upper flow in ML/Day",
                         style={'description_width': 'initial'},
                         layout={'color': 'black',
                                 'font-weight': 'bold'})])

buffer_label = HBox([Label(value="4. Set buffer around location",
                           style={'description_width': 'initial'})])

location_label = HBox([Label(value='5. location data source (select one):',
                             style={'description_width': 'initial'})])

gauge_label = HBox([Label(value='6. Enter gauge number:',
                          style={'description_width': 'initial'})])

slc_label = HBox([Label(value='7. Include or exclude L7 with slc failure:',
                        style={'description_width': 'initial'})])

cloud_label = HBox([Label(value='8. Set the maximum acceptable cloud percentage (per image):',
                          style={'description_width': 'initial'})])

export_label = HBox([Label(value='9. Set the export variables:',
                           style={'description_width': 'initial'})])

#Dashboard:
iht_dashboard = Tab()
iht_dashboard.children = [VBox([
    deploy_env_label,
    deploy_environment,
    date_selection_label,
    HBox([start_date, end_date]),
    flow_label,
    HBox([min_flow, max_flow]),
    buffer_label,
    buffer_value,
    location_label,
    use_gauge_location,
    use_my_own_location,
    upload_shapefile,
    gauge_label,
    gauge_num,
    slc_label,
    slc_option,
    cloud_label,
    cloud_threshold,
    export_label,
    export_variables
])]

iht_dashboard.set_title(0, 'IHT Inputs')


class IhtValues:
    deploy_environment = None
    start_date: datetime.datetime = None
    end_date: datetime.datetime = None
    gauge_num: str = None
    min_flow: int = None
    max_flow: int = None
    band_option: str = None
    water_threshold: float = None

    export_images: bool = None
    export_summary_images: bool = None
    cloud_threshold: float = None

    check_gauge: bool = None
    check_own: bool = None
    check_shapefile: bool = None

    slc_option: str = None

    input_lon: float = None
    input_lat: float = None

    shapefile_name: str = None
    
    def __init__(self) -> None:
        self.deploy_environment = copy.deepcopy(deploy_environment.value)
        
        self.start_date = copy.deepcopy(start_date.value)
        self.end_date = copy.deepcopy(end_date.value)
        self.gauge_num = copy.deepcopy(gauge_num.value)

        self.min_flow = copy.deepcopy(min_flow.value)
        self.max_flow = copy.deepcopy(max_flow.value)
        self.band_option = copy.deepcopy(band_option.value)
        self.summary_band_option = copy.deepcopy(summary_band_option.value)
        self.water_threshold = copy.deepcopy(water_threshold.value)

        self.export_images = copy.deepcopy(export_images.value)
        self.export_summary_images = copy.deepcopy(export_summary_images.value)
        self.cloud_threshold = copy.deepcopy(cloud_threshold.value)

        self.check_gauge = copy.deepcopy(check_gauge.value)
        self.check_own = copy.deepcopy(check_own.value)

        self.check_shapefile = copy.deepcopy(check_shapefile.value)
        
        self.slc_option = copy.deepcopy(slc_option.value)

        self.input_lon: copy.deepcopy(input_lon.value)
        self.input_lat: copy.deepcopy(input_lat.value)

        self.shapefile_loc = copy.deepcopy(shapefile_loc.value)
        
        self.errors = self.validate()
        self.valid = not self.errors

    def __str__(self) -> str:
        s = {}
        for k, v in self.__dict__.items():
            if not k.startswith("_"):
                s[k] = v
        return pprint.pformat(s, indent=4)

    def validate(self) -> List[str]:
        errors = []
        if not self.start_date:
            errors.append(f"You must enter a value for 'Start Date'")
        if not self.end_date:
            errors.append(f"You must enter a value for 'End Date'")
        if sum([check_gauge.value, check_own.value, check_shapefile.value]) != 1:
            errors.append(
                f"You must pick exactly ONE of ('{check_gauge.description}', "
                f"'{check_own.description}', '{check_shapefile.description}')."
            )
        return errors

    def get_clean_gauge(self) -> str:
        return self.gauge_num.clean()


def get_coords() -> Tuple[float, float, float, float]:
    ''' Takes in user selections for location
    returns the bounds of the area in lat and lon'''
    buffer = buffer_value.value

    if check_gauge.value or check_own.value:
        if check_gauge.value:
            lat, lon = get_gauge_coords()
        else: # Check Own
            lat = input_lat.value
            lon = input_lon.value
        lat_low = round((lat - buffer), 2)
        lat_high = round((lat + buffer), 2)
        lon_low = round((lon - buffer), 2)
        lon_high = round((lon + buffer), 2)

    if check_shapefile.value:
        vector_file = shapefile_loc.value
        gdf = gpd.read_file(vector_file) # -> GeoDataFrame
        # https://geopandas.org/docs/reference/api/geopandas.GeoDataFrame.html
        bounds = gdf.bounds
        lon_low = float(round(gdf.bounds.minx, 2)[0])
        lon_high = float(round(gdf.bounds.maxx, 2)[0])
        lat_low = float(round(gdf.bounds.miny, 2)[0])
        lat_high = float(round(gdf.bounds.maxy, 2)[0])
    return lat_low, lat_high, lon_low, lon_high


def get_gauge_coords() -> Tuple[float, float]:
    df = pd.read_csv('gauge_data/bom_gauge_data.csv')
    lat = float(df.loc[df['gauge number'][df['gauge number'] == str(gauge_num.value)].index]['lat'])
    lon = float(df.loc[df['gauge number'][df['gauge number'] == str(gauge_num.value)].index]['lon'])
    return lat, lon
