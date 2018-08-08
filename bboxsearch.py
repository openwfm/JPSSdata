from cmr import CollectionQuery, GranuleQuery
import json

api = GranuleQuery()

#MOD14: bounding box = Colorado (gps coord sorted lat,lon; counterclockwise)
MOD14granules = api.parameters(
                        short_name="MOD14",
                        platform="Terra",
                        downloadable=True,
                        polygon=[(-109.0507527,40.99898), (-109.0698568,37.0124375), (-102.0868788,36.9799819),(-102.0560592,40.999126), (-109.0507527,40.99898)],
                        temporal=("2016-10-10T01:02:00Z", "2016-10-12T00:00:30Z") #time start,end
                        )
print MOD14granules.hits()
MOD14granules = api.get(10)

for granule in MOD14granules:
    print granule["title"], granule["time_start"], granule["time_end"], granule["polygons"]