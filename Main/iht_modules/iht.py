# +
# %matplotlib inline
import ipywidgets as widgets
import geopandas as gpd

# Input widgets #

# flow and buffer
min_flow = widgets.BoundedIntText(
    value=0,
    min=0,
    max=999999,
    step=1,
    description='Minimum flow:',
    disabled=False)

max_flow = widgets.BoundedIntText(
    value=0,
    min=0,
    max=999999,
    step=1,
    description='Maximum flow:',
    disabled=False)
    
buffer = widgets.BoundedFloatText(
    value=0.2,
    min=0,
    max=1,
    step=0.01,
    description='Buffer around location:',
    disabled=False,
    style = {'description_width': 'initial'}
) 

#Gauge option
check_gauge = widgets.Checkbox(
    value=False,
    description='Use gauge location',
    disabled=False,
    indent=False,
    style = {'description_width': 'initial'}
)

use_gauge_location = widgets.Accordion(children=[check_gauge])

# user option 
check_own = widgets.Checkbox(
    value=False,
    description='Define my own location',
    disabled=False,
    indent=False,
    style = {'description_width': 'initial'}
)
input_lat = widgets.BoundedFloatText(
    value=0.0,
    min=-99999999999999,
    max=999999999999999,
    step=0.0001,
    description='Latitude:',
    disabled=False
)

input_lon = widgets.BoundedFloatText(
    value=0.0,
    min=-99999999999999,
    max=9999999999999,
    step=0.0001,
    description='Longitude:',
    disabled=False
)

use_my_own_location = widgets.Accordion(children=[widgets.VBox([check_own,input_lat,input_lon])])

#UPload shapefile:
check_shapefile = widgets.Checkbox(
    value=False,
    description='Use shapefile from shapefile_input folder:',
    disabled=False,
    indent=False,
    style = {'description_width': 'initial'}
)

shapefile_loc = widgets.Text(value='shapefile_inputs/',
                         placeholder='Enter the name of the shapefile in the folder',
                         description='Name of shapefile:',
                         disabled=False,
                         style = {'description_width': 'initial'}
                        )

upload_shapefile = widgets.Accordion(children=[widgets.VBox([check_shapefile, shapefile_loc])])

# Labels:
use_gauge_location.set_title(0, 'Use coordinates for the gauge to define extents')
use_my_own_location.set_title(0, 'Input coordinates to define extents')
upload_shapefile.set_title(0, 'Upload shapefile to define extents')

flow_label = widgets.HBox([widgets.Label(value="1. Set lower and upper flow in ML/Day",
                                        style = {'description_width': 'initial'},
                                        layout = {'color': 'black',
                                                'font-weight': 'bold'})])

buffer_label = widgets.HBox([widgets.Label(value="2. Set buffer around location",
                                            style = {'description_width': 'initial'})])

location_label = widgets.HBox([widgets.Label(value= '3. location data source (select one):',
                                            style = {'description_width': 'initial'})])

#Dashboard:
iht_dashboard = widgets.Tab()
iht_dashboard.children = [widgets.VBox([flow_label, min_flow, max_flow, buffer_label, buffer, 
                         location_label,
                                        use_gauge_location, use_my_own_location, upload_shapefile])]
iht_dashboard.set_title(0, 'IHT Inputs')

#Function to retrived the data
def get_flow_bounds(user_input_min, user_input_max):
    ''' Returns flow bounds depending on user selection '''
    
    y_low = user_input_min
    y_high = user_input_max
    
    return y_low, y_high
    
def get_coords(gauge_input, self_input, user_lat, user_lon, shape_input, shapefile_path,
              buffer, station_loc):
    ''' Takes in user selections for location 
    returns the bounds of the area in lat and lon'''
    
    if sum([gauge_input, self_input, shape_input]) > 1:
        return 'ERROR: More than one location source defined'
    if sum([gauge_input, self_input, shape_input]) == 0:
        return 'ERROR: No location source box selected'
    
    if gauge_input == True:
            
        lat, lon = station_loc
        lat_low = round((lat - buffer), 2)
        lat_high = round((lat + buffer), 2)
        lon_low = round((lon - buffer), 2)
        lon_high = round((lon + buffer), 2)
        
        return lat_low, lat_high, lon_low, lon_high
        
    if self_input == True:
        lat = user_lat
        lon = user_lon
        lat_low = round((lat - buffer), 2)
        lat_high = round((lat + buffer), 2)
        lon_low = round((lon - buffer), 2)
        lon_high = round((lon + buffer), 2)
        
        return lat_low, lat_high, lon_low, lon_high
    
    if shape_input == True:
        try:
            vector_file = shapefile_path
            gdf = gpd.read_file(vector_file)
            lon_low = round(gdf.bounds.minx,2)
            lon_high = round(gdf.bounds.maxx,2)
            lat_low = round(gdf.bounds.miny,2)
            lat_high = round(gdf.bounds.maxy, 2)
        except DriverError:
            print('No such file exists, check the file name and run this cell again')
        
        return lat_low, lat_high, lon_low, lon_high



