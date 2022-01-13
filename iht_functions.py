# -*- coding: utf-8 -*-
import time
import logging
from datetime import datetime
from typing import Optional, Dict, List
import pandas as pd
import ee
import xarray as xr
import plotly.graph_objects as go
import folium


logging.basicConfig()
log = logging.getLogger(__name__[:-3])
log.setLevel(logging.INFO)

################################################################################################
# ## Google Earth Engine Satelllite Imagery Processing ###

def bands8(img: ee.image.Image) -> ee.image.Image:
    '''Function to change landsat 8 bandnames'''
    return img.select('SR_B2','SR_B3','SR_B4','SR_B5','SR_B6','SR_B7') \
              .rename('blue', 'green', 'red', 'nir', 'swir1', 'swir2')


def bands57(img: ee.image.Image) -> ee.image.Image:
    '''Function to change landsat 5 and 7 bandnames'''
    return img.select('SR_B1','SR_B2','SR_B3','SR_B4','SR_B5','SR_B7') \
              .rename('blue', 'green', 'red', 'nir', 'swir1', 'swir2')


def bandsS2(img: ee.image.Image) -> ee.image.Image:
    '''Function to change Sentinel 2 bandnames'''
    return img.select('B2', 'B3', 'B4', 'B8', 'B11', 'B12') \
              .rename('blue', 'green', 'red', 'nir', 'swir1', 'swir2')


def maskQuality(image: ee.image.Image) -> ee.element.Element:
    ''' Function to quality mask from the QA_PIXEL bits of Landsat Collection 2 data 
        Bit 0 - Fill
        Bit 1 - Dilated Cloud
        Bit 2 - Unused
        Bit 3 - Cloud
        Bit 4 - Cloud Shadow
    '''
    qaMask = image.select('QA_PIXEL').bitwiseAnd(int('11111', base=2)).eq(0)
    saturationMask = image.select('QA_RADSAT').eq(0)

    # Apply the scaling factors to the appropriate bands.
    opticalBands = image.select('SR_B.').multiply(0.0000275).add(-0.2)
    thermalBands = image.select('ST_B.*').multiply(0.00341802).add(149.0)

    # Replace the original bands with the scaled ones and apply the masks.
    return image.addBands(srcImg = opticalBands, overwrite = True)\
        .addBands(srcImg = thermalBands, overwrite = True)\
        .updateMask(qaMask)\
        .updateMask(saturationMask)

def maskS2clouds(image):
    '''Function to mask clouds using the Sentinel-2 QA band.'''
    qa = image.select('QA60')

    # Bits 10 and 11 are clouds and cirrus, respectively.
    cloudBitMask = 1 << 10
    cirrusBitMask = 1 << 11

    # Both flags should be set to zero, indicating clear conditions.
    mask = qa.bitwiseAnd(cloudBitMask).eq(0).And(
        qa.bitwiseAnd(cirrusBitMask).eq(0))

    # Mask image
    image = image.unmask()\
                 .mask(mask)\
                 .copyProperties(image, ["system:time_start"])

    # Return the masked and scaled data, without the QA bands.
    return image
                

def clipEdges(img):
    ''' Function to clip edges of Landsat'''
    return img.clip(img.geometry().buffer(-6000))

def unQA(image):
    ''' Function to unmask masked Sentinel-2 surface reflectance'''
    return image.unmask(0)

def addNDWI(image):
    ndwi = image.normalizedDifference(['green', 'nir']).rename('NDWI')
    return image.addBands(ndwi)

def addMNDWI(image):
    mndwi = image.normalizedDifference(['green', 'swir1']).rename('MNDWI')
    return image.addBands(mndwi)

def addFisher(image): # FYI Fisher index optimum threshold 0.63
    # Define an Array of Tasseled Cap coefficients.
    coefficients = ee.Array([[1.7204, 171, 3, -70, -45, -71]])

    # Make an Array Image, with a 1-D Array per pixel.
    constant = ee.Image(1)
    arrayImage1D = image.addBands(constant).select(
        ['constant', 'green', 'red', 'nir', 'swir1', 'swir2']).toArray()

    # Make an Array Image with a 2-D Array per pixel, 6x1.
    arrayImage2D = arrayImage1D.toArray(1)

    # Do a matrix multiplication: 6x6 times 6x1.
    componentsImage = ee.Image(coefficients) \
                        .matrixMultiply(arrayImage2D) \
                        .arrayProject([0]) \
                        .arrayFlatten([['Fisher']])
    return image.addBands(componentsImage.rename('Fisher'))

def add_ee_layer(foliumMap: folium.folium.Map, ee_image_object: ee.image.Image,
                 vis_params: Dict[str, List[str]], name: str) -> None:
    '''Define a method for displaying Earth Engine image tiles to folium map.

    Parameters
    ----------
    foliumMap: folium.folium.Map
               The name of the folium map object
    ee_image_object: ee.image.Image
                     The ee image object
    vis_params: Dict[str, List[str]]
                The visualization parameters as a (client-side) JavaScript object. See
                https://developers.google.com/earth-engine/apidocs/ee-data-getmapid
    name : str
           The name of the ee layer
    '''
    map_id_dict = ee.Image(ee_image_object).getMapId(vis_params)
    folium.raster_layers.TileLayer(
        tiles=map_id_dict['tile_fetcher'].url_format,
        attr="Map Data © Google Earth Engine",
        name=name,
        overlay=True,
        control=True
    ).add_to(foliumMap)

###################################################################################################
# ## Gauge and date processing ###

def getTime(imcol: ee.imagecollection.ImageCollection) -> Optional[xr.DataArray]:
    try:
        # What dates do we have
        acq_times = imcol.aggregate_array('system:time_start').getInfo()
        acq_times2 = [time.strftime("%d/%m/%Y, %H:%M:%S", time.gmtime(acq_time/1000))
                      for acq_time in acq_times]
        acq_times2 = [datetime.strptime(acq_time, "%d/%m/%Y, %H:%M:%S") for acq_time in acq_times2]
        time_arr = xr.DataArray(data=acq_times2, dims='time')
        time_arr = time_arr.assign_attrs(units='seconds since 1901-01-01 00:00:00')
        time_arr = time_arr.assign_coords(time=('time', acq_times2))
        time_arr['time'] = time_arr.indexes['time'].normalize()
        time_arr = time_arr.sortby('time').resample(time='1D').first().dropna('time')
        return time_arr
    except Exception:
        log.warning('A dry exception block occurred, please review this code and handle the '
                    'exception gracefully by checking the validity of input parameters.')
        raise

def countPasses(gee_times: Optional[xr.DataArray]) -> str:
    try:
        return str(gee_times.shape[0])
    except Exception:
        log.warning('A dry exception block occurred, please review this code and handle the '
                    'exception gracefully by checking the validity of input parameters.')
        raise


def gauge_data_cleaner(input_data: pd.core.frame.DataFrame) -> pd.core.frame.DataFrame:
    '''Ingests a pd.df of flow timeseries,
    converts data types and measurements,
    returns clean gauge data'''
    gauge_df = input_data.copy(deep=True)
    gauge_df.drop(['DATASOURCEID', 'SITEID', 'SUBJECTID', 'QUALITYCODE'], axis=1, inplace=True)
    gauge_df.rename(columns={'DATETIME': 'Timestamp', 'VALUE': 'Value'}, inplace=True)
    gauge_df["Value"] = pd.to_numeric(gauge_df["Value"], downcast="float")
    gauge_df['Timestamp'] = pd.to_datetime(gauge_df['Timestamp'], format='%Y-%m-%d')
    gauge_df = gauge_df.set_index('Timestamp')
    return gauge_df

def merge_satellite_with_gauge(gee_times: Optional[xr.DataArray],
                               gauge_input: pd.core.frame.DataFrame, flow_band: Dict[str, int]):
    '''Ingests gauge data and satellite data xr,
    identifies where on the gauge data there are passes and works out how many fall within the
    flow bands of interest,
    returns the dates, how many there are, and the xarray of the merged data
    '''
    try:
        gauge_data_xr = gauge_input.drop_duplicates().to_xarray()
        merged_data = gauge_data_xr.interp(Timestamp=gee_times) # problem line
        specified_satellite_passes = merged_data.where((
            (merged_data.Value > flow_band['min flow']) &\
            (merged_data.Value < flow_band['max flow'])),
                                                       drop=True)
        specified_satellite_passes = specified_satellite_passes.drop('Timestamp')
        how_many = gee_times.shape[0]
        return how_many, specified_satellite_passes
    except Exception:
        log.warning('A dry exception block occurred, please review this code and handle the '
                    'exception gracefully by checking the validity of input parameters.')
        raise


def convert_to_pandas(input_merged_data, input_specified_passes):
    ''' Converts to dataframe
    returns the dataframe with the gauge reading at the time of the satellite pass '''
    try:
        merged_data_pd = input_merged_data.to_dataframe()

        all_specified_passes_pd = input_specified_passes.time.to_dataframe()
        all_specified_passes_pd = all_specified_passes_pd.rename(columns={'time': 'date'})
        all_merged_data = pd.merge(all_specified_passes_pd, merged_data_pd, left_on='time',
                                   right_index=True, how='inner')
        all_merged_data = all_merged_data.drop(columns='date')
        all_merged_data.index = all_merged_data.index.normalize()
        all_merged_data.index.name = 'date'
        return all_merged_data
    except Exception:
        log.warning('A dry exception block occurred, please review this code and handle the '
                    'exception gracefully by checking the validity of input parameters.')
        raise

def add_flow_bands_to_gauge_data(input_gauge_data, flow_band):
    ''' Adds the flow bands of interest to the gauge dataframe for graphing purposes '''
    try:
        graph_gauge_df = input_gauge_data.copy(deep=True)
        graph_gauge_df['y_lower'] = flow_band['min flow']
        graph_gauge_df['y_higher'] = flow_band['max flow']
        return graph_gauge_df
    except Exception:
        log.warning('A dry exception block occurred, please review this code and handle the '
                    'exception gracefully by checking the validity of input parameters.')
        raise

def rising_falling_main(multiplier: int, days_ahead: int, input_gauge_df: pd.core.frame.DataFrame,
                        input_sat_passes):
    ''' Split the satellite passes into either the rising or falling category '''
    try:
        rising_list = list()
        falling_list = list()

        for i, flow in enumerate(input_gauge_df['Value'][:len(input_gauge_df['Value'])-days_ahead]):
            if flow < input_gauge_df['Value'][i + days_ahead] * multiplier:
                rising_list.append(input_gauge_df.index[i])
            else:
                falling_list.append(input_gauge_df.index[i])

        rising_passes = list(set(rising_list) & set(list(input_sat_passes.index)))
        falling_passes = list(set(falling_list) & set(list(input_sat_passes.index)))

        rising_passes_df = df_constructor(rising_passes, input_sat_passes)
        falling_passes_df = df_constructor(falling_passes, input_sat_passes)
        return rising_passes_df, falling_passes_df
    except Exception:
        log.warning('A dry exception block occurred, please review this code and handle the '
                    'exception gracefully by checking the validity of input parameters.')
        raise

def df_constructor(input_list, sat_data_to_join):
    ''' converts the list of either rising or falling passes into a dataframe '''
    df = pd.DataFrame(input_list, columns=['date'])
    df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
    df = df.set_index('date')
    df = df.join(sat_data_to_join)
    return df

def rising_falling_cleaner(input_df, sat_source, pass_or_fail):
    ''' Ingests rising or falling dataframe,
    drops the columns not required for the rest of the analysis,
    adds column for the satellite and whether or not it will be rejected/passed'''
    try:
        df = input_df.copy(deep=True)
        df['sat_result'] = f'{sat_source} {pass_or_fail}'
        return df
    except Exception:
        log.warning('A dry exception block occurred, please review this code and handle the '
                    'exception gracefully by checking the validity of input parameters.')
        raise

def switch(row):
    ''' A function to help switch accept and reject in Google Colab'''
    if not row['Change']:
        return row
    if row['sat_result'] == 'landsat reject':
        row['sat_result'] = 'landsat accept'
        row['Change'] = False
    elif row['sat_result'] == 'landsat accept':
        row['sat_result'] = 'landsat reject'
        row['Change'] = False
    elif row['sat_result'] == 'sentinel reject':
        row['sat_result'] = 'sentinel accept'
        row['Change'] = False
    elif row['sat_result'] == 'sentinel accept':
        row['sat_result'] = 'sentinel reject'
        row['Change'] = False
    return row

##################################################################################################
# ## Graphing functions ###

def graph_all(flow_band: Dict[str, int], input_gauge_data: pd.core.frame.DataFrame, input_ls, input_s):

    ''' Function to generate a graph showing the satellite passes relative to where they occur
    on the hydrograph '''

    # Set up the dataframe:
    graph_gauge_df = add_flow_bands_to_gauge_data(input_gauge_data, flow_band)

    fig_all = go.Figure()

    # Add gauge data:
    fig_all.add_trace(go.Scatter(name='Gauge data',
                                 x=graph_gauge_df.index,
                                 y=graph_gauge_df['Value'],
                                 mode='lines',
                                 line=dict(color='royalblue', width=1),
                                 hovertemplate='flow: %{y:.2f}<extra></extra>' +
                                 '<br><b>Date</b>: %{x}<br>'))
    # Add landsat:
    try:
        # Add landsat:
        fig_all.add_trace(go.Scatter(name='Landsat pass',
                                     x=input_ls.index,
                                     y=input_ls['Value'],
                                     mode='markers',
                                     line=dict(color='green', width=2),
                                     hovertemplate='Landsat pass <extra></extra>' +
                                     '<br><b>Date</b>: %{x}<br>',
                                     marker_size=9))
    except Exception:
        log.warning('A dry exception block occurred, please review this code and handle the '
                    'exception gracefully by checking the validity of input parameters '
                    'for Landsat pass.')

    # Add sentinel:
    try:
        fig_all.add_trace(go.Scatter(name='Sentinel pass',
                                     x=input_s.index,
                                     y=input_s['Value'],
                                     mode='markers',
                                     line=dict(color='deepskyblue', width=2),
                                     hovertemplate='Sentinel pass <extra></extra>' +
                                     '<br><b>Date</b>: %{x}<br>',
                                     marker_symbol='cross',
                                     marker_size=9))
    except Exception:
        log.warning('A dry exception block occurred, please review this code and handle the '
                    'exception gracefully by checking the validity of input parameters '
                    'for Sentinel pass.')
    # Adding the lower flow bounds:
    fig_all.add_trace(go.Scatter(name='lower flow bound',
                                 x=graph_gauge_df.index,
                                 y=graph_gauge_df['y_lower'],
                                 mode='lines',
                                 line=dict(color='black', width=2)))
    # Adding the upper flow bounds:
    fig_all.add_trace(go.Scatter(name='upper flow bound',
                                 x=graph_gauge_df.index,
                                 y=graph_gauge_df['y_higher'],
                                 mode='lines',
                                 line=dict(color='black', width=2)))
    fig_all.update_layout(hovermode="closest",
                          title='Landsat and Sentinel passes within the flow band of interest',
                          xaxis_title='Date',
                          yaxis_title='Flow at gauge (ML/Day)',
                          font=dict(family='Calibri', size=16, color='black'),
                          paper_bgcolor='white',
                          plot_bgcolor='white',
                          hoverlabel=dict(
                              bgcolor="white",
                              font_size=16,
                              font_family="Rockwell"))
    return fig_all

def graph_rising_falling(flow_band, input_gauge_data, input_rising_ls, input_falling_ls,
                         input_rising_s=None, input_falling_s=None):
    ''' Function to generate a graph showing the satellite passes relative to where they occur
    on the hydrograph and their respective category (either rising or falling)'''
    if input_rising_ls is None:
        input_rising_ls = []
    if input_falling_ls is None:
        input_falling_ls = []
    graph_gauge_df = add_flow_bands_to_gauge_data(input_gauge_data, flow_band)

    fig_rise_fall = go.Figure()

    fig_rise_fall.add_trace(go.Scatter(name='Gauge data',
                                       x=graph_gauge_df.index,
                                       y=graph_gauge_df['Value'],
                                       mode='lines',
                                       line=dict(color='royalblue', width=1),
                                       hovertemplate=('flow: %{y:.2f}<extra></extra>'
                                                      '<br><b>Date</b>: %{x}<br>')))


    try:
        fig_rise_fall.add_trace(go.Scatter(name='Rising Landsat pass',
                                           x=input_rising_ls.index,
                                           y=input_rising_ls['Value'],
                                           mode='markers',
                                           marker_symbol='triangle-up',
                                           marker_size=9,
                                           line=dict(color='green', width=2),
                                           hovertemplate='Rising Landsat pass <extra></extra>' +
                                           '<br><b>Date</b>: %{x}<br>'))
    except Exception:
        log.warning('A dry exception block occurred, please review this code and handle the '
                    'exception gracefully by checking the validity of input parameters.')
        raise
    try:
        fig_rise_fall.add_trace(go.Scatter(name='Falling Landsat pass',
                                           x=input_falling_ls.index,
                                           y=input_falling_ls['Value'],
                                           mode='markers',
                                           marker_symbol='triangle-down',
                                           marker_size=9,
                                           line=dict(color='crimson', width=2),
                                           hovertemplate='Falling Landsat pass <extra></extra>' +
                                           '<br><b>Date</b>: %{x}<br>'))
    except Exception:
        log.warning('A dry exception block occurred, please review this code and handle the '
                    'exception gracefully by checking the validity of input parameters.')
        raise

    try:
        fig_rise_fall.add_trace(go.Scatter(name='Rising Sentinel pass',
                                           x=input_rising_s.index,
                                           y=input_rising_s['Value'],
                                           mode='markers',
                                           marker_symbol='star-triangle-up',
                                           marker_size=9,
                                           line=dict(color='deepskyblue', width=2),
                                           hovertemplate='Rising Sentinel pass <extra></extra>' +
                                           '<br><b>Date</b>: %{x}<br>'))
        print("{} rising sentinel passes".format(len(input_rising_s)))
    except Exception:
        log.warning('A dry exception block occurred, please review this code and handle the '
                    'exception gracefully by checking the validity of input parameters.')
        raise
    try:
        fig_rise_fall.add_trace(go.Scatter(name='Falling Sentinel pass',
                                           x=input_falling_s.index,
                                           y=input_falling_s['Value'],
                                           mode='markers',
                                           marker_symbol='star-triangle-down',
                                           marker_size=9,
                                           line=dict(color='lightpink', width=2),
                                           hovertemplate='Falling Sentinel pass <extra></extra>' +
                                           '<br><b>Date</b>: %{x}<br>'))
        print("{} falling sentinel passes".format(len(input_falling_s)))
    except Exception:
        log.warning('A dry exception block occurred, please review this code and handle the '
                    'exception gracefully by checking the validity of input parameters.')
        raise

    # Adding the lower flow bounds:
    fig_rise_fall.add_trace(go.Scatter(name='lower flow bound',
                                       x=graph_gauge_df.index,
                                       y=graph_gauge_df['y_lower'],
                                       mode='lines',
                                       line=dict(color='black', width=2)))

    # Adding the upper flow bounds:
    fig_rise_fall.add_trace(go.Scatter(name='upper flow bound',
                                       x=graph_gauge_df.index,
                                       y=graph_gauge_df['y_higher'],
                                       mode='lines',
                                       line=dict(color='black', width=2)))

    fig_rise_fall.update_layout(hovermode='closest',
                                title='Sentinel and Landsat broken into rising/falling categories',
                                xaxis_title='Date',
                                yaxis_title='Flow at gauge (ML/Day)',
                                font=dict(family='Calibri', size=16, color='black'),
                                paper_bgcolor='white',
                                plot_bgcolor='white',
                                hoverlabel=dict(
                                    bgcolor='white',
                                    font_size=16,
                                    font_family='Rockwell'))

    print(f'{len(input_rising_ls)} rising landsat passes')
    print(f'{len(input_falling_ls)} falling landsat passes')
    return fig_rise_fall

def get_graph_object_color(x: str) -> str:
    '''Function to allocate a colour to the
        sat pass depending on how it is classified '''
    color_map = {
        'landsat accept': 'green',
        'landsat reject': 'crimson',
        'sentinel accept': 'deepskyblue',
    }
    if x not in color_map:
        return 'lightpink'
    return color_map[x]


class ScatterUpdatePoint:
    s_only = None
    ls_only = None
    master_df_reindex = None
    passing_failing_graph = None
    scatter = None
    
    def __init__(self, s_only, ls_only, master_df_reindex, passing_failing_graph, scatter):
        self.s_only = s_only
        self.ls_only = ls_only
        self.master_df_reindex = master_df_reindex
        self.passing_failing_graph = passing_failing_graph
        self.scatter = scatter

    def scatter_update_point(self, trace, points, selector):
        ''' Reclassify the points based on user selection '''
        try:
            if self.s_only:
                # TODO-MDBA, this is logically equivalent to previous code, I'm unsure if skipping
                # this function for s_only is intended.
                return
            # There is no valid Sentinel-2 data
            try:
                date = points.xs[0]
                row_name = self.master_df_reindex.loc[self.master_df_reindex['date'] == date]
                status = "accept"
                if date in self.ls_only.index:
                    point_type = "land_sat"
                    if row_name['sat_result'].iloc[0] == 'landsat accept':
                        status = "reject"
                for p in point_type:
                    self.master_df_reindex \
                        .loc[self.master_df_reindex['date'] == date, 'sat_result'] = f'{p} {status}'
                with self.passing_failing_graph.batch_update():
                    self.scatter.marker.color = list(map(get_graph_object_color,
                                                         self.master_df_reindex['sat_result']))
                print(f'The {" and ".join(point_type)} pass on {date} has been '
                      f'reclassified to be {status}ed')
            except IndexError:
                print('You missed the satellite pass! Try click again')
        except:
            # There is valid Sentinel-2 data
            try:
                date = points.xs[0]
                row_name = self.master_df_reindex.loc[self.master_df_reindex['date'] == date]
                status = "accept"
                point_type = []
                if date in self.ls_only.index and date in self.s_only.index:
                    point_type = ["landsat", "sentinel"]
                    if 'accept' in row_name['sat_result'].iloc[0]:
                        status = "reject"
                elif date in self.ls_only.index and date not in self.s_only.index:
                    point_type = ["landsat"]
                    if row_name['sat_result'].iloc[0] == 'landsat accept':
                        status = "reject"
                elif date in self.s_only.index and date not in self.ls_only.index:
                    point_type = ["sentinel"]
                    if row_name['sat_result'].iloc[0] == 'sentinel accept':
                        status = "reject"
                for p in point_type:
                    self.master_df_reindex \
                        .loc[self.master_df_reindex['date'] == date, 'sat_result'] = f'{p} {status}'
                with self.passing_failing_graph.batch_update():
                    self.scatter.marker.color = list(map(get_graph_object_color,
                                                         self.master_df_reindex['sat_result']))
                print(f'The {" and ".join(point_type)} pass on {date} has been '
                      f'reclassified to be {status}ed')
            except IndexError:
                print('You missed the satellite pass! Try click again')

def addAcceptRejectGraph(s_only, ls_only, master_df_reindex, clean_gauge_data, flow_band):
    ''' Interactive graph for toggling satellite passes to export '''

    graph_gauge_df = add_flow_bands_to_gauge_data(clean_gauge_data, flow_band)

    # Adding landsat trace:
    trace_sats = go.Scatter(name='Satellite pass',
                            x=master_df_reindex['date'],
                            y=master_df_reindex['Value'],
                            mode='markers',
                            marker=dict(size=8,
                                        color=list(map(get_graph_object_color,
                                                       master_df_reindex['sat_result']))),
                            showlegend=False)
    # Adding the gauge data:
    trace_gauge = go.Scatter(name='Gauge data',
                             x=graph_gauge_df.index,
                             y=graph_gauge_df['Value'],
                             mode='lines',
                             line=dict(color='royalblue', width=1),
                             hovertemplate='flow: %{y:.2f}<extra></extra>' +
                             '<br><b>Date</b>: %{x}<br>')
    # Adding the lower flow bounds:
    trace_lower = go.Scatter(name='lower flow bound',
                             x=graph_gauge_df.index,
                             y=graph_gauge_df['y_lower'],
                             mode='lines',
                             line=dict(color='black', width=2))
    # Adding the upper flow bounds:
    trace_upper = go.Scatter(name='upper flow bound',
                             x=graph_gauge_df.index,
                             y=graph_gauge_df['y_higher'],
                             mode='lines',
                             line=dict(color='black', width=2))
    # Adding for legend
    trace_pass_ls = go.Scatter(name='landsat accepted',
                               x=[None], y=[None],
                               mode='markers',
                               marker=dict(size=8, color='green'),
                               showlegend=True)
    # Adding for legend
    trace_fail_ls = go.Scatter(name='landsat rejected',
                               x=[None], y=[None],
                               mode=' markers',
                               marker=dict(size=8, color='crimson'),
                               showlegend=True)
    try:
        trace_pass_s = go.Scatter(name='sentinel accepted',
                                  x=[None], y=[None],
                                  mode='markers',
                                  marker=dict(size=8, color='deepskyblue'),
                                  showlegend=True)
    except Exception:
        log.warning('A dry exception block occurred, please review this code and handle the '
                    'exception gracefully by checking the validity of input parameters.')
        raise

    try:
        # Adding for legend
        trace_fail_s = go.Scatter(name='sentinel rejected',
                                  x=[None], y=[None],
                                  mode=' markers',
                                  marker=dict(size=8, color='lightpink'),
                                  showlegend=True)
    except Exception:
        log.warning('A dry exception block occurred, please review this code and handle the '
                    'exception gracefully by checking the validity of input parameters.')
        raise

    # Add the traces to the figure:
    passing_failing_graph = go.FigureWidget(data=[trace_sats, trace_gauge,
                                                  trace_lower, trace_upper,
                                                  trace_pass_ls, trace_fail_ls,
                                                  trace_pass_s, trace_fail_s])

    passing_failing_graph.update_layout(hovermode="closest",
                                        title='Proposed satellite passes to use in the analysis',
                                        xaxis_title='Date',
                                        yaxis_title='Flow at gauge (ML/Day)',
                                        font=dict(family='Calibri', size=16, color='black'),
                                        paper_bgcolor='white',
                                        plot_bgcolor='white',
                                        hoverlabel=dict(
                                            bgcolor="white",
                                            font_size=16,
                                            font_family="Rockwell"))

    scatter = passing_failing_graph.data[0]

    scatter_point = ScatterUpdatePoint(s_only, ls_only, master_df_reindex,
                                       passing_failing_graph, scatter)
    scatter.on_click(scatter_point.scatter_update_point)
    return passing_failing_graph
