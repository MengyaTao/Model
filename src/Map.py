import pandas as pd
import plotly.plotly as py
import plotly.offline as offline
import plotly.figure_factory as ff

py.sign_in('VenkatRamanf531', 'rp3EIEo9M7GfjrHx9cUx')

# plot the whole US

fips = ['06021', '06023', '06027',
        '06029', '06033', '06059',
        '06047', '06049', '06051',
        '06055', '06061']
values = range(len(fips))
fig = ff.create_choropleth(fips=fips, values=values)
# offline.plot(fig, filename='test', image='png')
py.image.save_as(fig, filename='test.png') # but this only works on line



'''
- install basemap packages
    : https://stackoverflow.com/questions/42299352/installing-basemap-on-mac-python
    : https://matplotlib.org/basemap/users/installing.html
    : brew install gets; brew install geos
    : or conda install basemap

- use Plotly: https://plot.ly/python/county-choropleth/
    : pip install geopandas==0.3.0
    : pip install pyshp==1.2.10
    : pip install shapely==1.6.3

'''