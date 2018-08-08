from cmr import CollectionQuery, GranuleQuery
import json
api = GranuleQuery()

#MOD14: data - auto-sorted by start_time
MOD14granules = api.parameters(
                        short_name="MOD14",
                        platform="Terra",
                        downloadable=True)

print MOD14granules.hits()
#MOD14granules = api.get_all()
MOD14granules = api.get(10)

for granule in MOD14granules:
    print granule["title"], granule["time_start"], granule["time_end"], granule["polygons"]

#MOD03: geolocation data - auto-sorted by start_time
MOD03granules = api.parameters(
                        short_name="MOD03",
                        platform="Terra",
                        downloadable=True)

print MOD03granules.hits()
MOD03granules = api.get(10)

for granule in MOD03granules:
    print granule
    #print granule["title"], granule["time_start"], granule["time_end"], granule["longitude"]
    #"polygons" gives long/lat data in pairs-first/last pair are the same
    # note - it appears that all granules are quadrilaterals