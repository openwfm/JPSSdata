from osgeo import ogr
from shapely.geometry import MultiLineString
import json, sys, math
import os.path as osp

def contour2shp(data, shp_path, geot, projection):
	suffix = 'TIGN_G'
	shpfiles = []
	for idx, c in enumerate(data['contours']):
		if len(c['polygons'])>0:
			time = c['time_begin'].replace('T','_')[:-1]
			newfilename = osp.join(shp_path, suffix+'_'+time+'.shp')
			print('> creating shp file: %s' % newfilename)
			# Multiline object
			multi = MultiLineString(c['polygons'])
			# Now convert it to a shapefile with OGR
			driver = ogr.GetDriverByName('Esri Shapefile')
			dataset = driver.CreateDataSource(newfilename)
			layer = dataset.CreateLayer('', srs = projection, geom_type = ogr.wkbMultiLineString)
			# Add one attribute
			layer.CreateField(ogr.FieldDefn('id', ogr.OFTInteger))
			defn = layer.GetLayerDefn()
			# Create a new feature (attribute and geometry)
			feat = ogr.Feature(defn)
			feat.SetField('id', 123)
			# Make a geometry, from Shapely object
			geom = ogr.CreateGeometryFromWkb(multi.wkb)
			feat.SetGeometry(geom)
			layer.CreateFeature(feat)
			# Save and close everything
			shpfiles.append(newfilename)
			ds = layer = feat = geom = None

	return shpfiles

