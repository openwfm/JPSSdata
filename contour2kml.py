import json, sys, math

def contour2kml(data,kml_path):
   with open(kml_path,'w') as kml:
       name = data.get('name','No Name')
       kml.write("""<?xml version="1.0" encoding="UTF-8"?>\n""")
       kml.write("""<kml xmlns="http://www.opengis.net/kml/2.2" xmlns:gx="http://www.google.com/kml/ext/2.2" xmlns:kml="http://www.opengis.net/kml/2.2" xmlns:atom="http://www.w3.org/2005/Atom">\n""")
       kml.write("""<Document><name>%s</name>\n""" % name)
       # write all styles first
       print data
       for idx, c in enumerate(data['contours']):
           kml.write("""<Style id="ColorStyle%s">""" % idx)
           if 'text' in c and c['text'] is not None:
               kml.write("""
                <BalloonStyle>
                        <text>%s</text>
                </BalloonStyle>""" % c['text'])
           if 'LineStyle' in c and c['LineStyle'] is not None:
                kml.write("""
                <LineStyle>
                        <color>%s</color>
                        <width>%s</width>
                </LineStyle>""" % (c['LineStyle']['color'],c['LineStyle'].get('width',3)))
           if 'PolyStyle' in c and c['PolyStyle'] is not None:
               kml.write("""
                <PolyStyle>
                        <color>%s</color>
                        <colorMode>%s</colorMode>
                </PolyStyle>""" % (c['PolyStyle']['color'],c['PolyStyle'].get('colorMode','random')))
           kml.write("\n</Style>")
       folder_name = data.get('folder_name',name)
       kml.write("""
            <Folder><name>%s</name>
		<open>1</open>""" % folder_name)
       for idx, c in enumerate(data['contours']):
           time_begin = c['time_begin']
           kml.write("""
		<Folder id="layer 0">
			<name>%s</name>
			<Placemark>
                                <TimeSpan><begin>%s</begin></TimeSpan>
				<name>%s</name>
				<styleUrl>ColorStyle%s</styleUrl>
				<MultiGeometry>""" % (
               time_begin, time_begin, time_begin,idx))
           for polygon in c['polygons']: 
               kml.write("""
					<Polygon>
						<outerBoundaryIs>
							<LinearRing>
								<coordinates>""")
               for segment in polygon:
                   kml.write("\n%s,%s,0" % tuple(segment))
                   kml.write("""
                                                                </coordinates>
                                                        </LinearRing>
                                                </outerBoundaryIs>
                                        </Polygon>""")
           kml.write("""
				</MultiGeometry>
			</Placemark>
		</Folder>""")
       kml.write("""
       </Folder>
</Document>
</kml>
"""  )

if __name__ == '__main__':
    print 'Running a self-test case'
    data={
         'name':'selftest.kmz',
         'folder_name':'Test Perimeters',
         'contours': [
             {'text':'2011-06-28T23:43:00-06:00',
              'LineStyle':{
                  'color':'ff081388',
                  'width':'2.5',
              },
              'PolyStyle':{
                  'color':'66000086',
                  'colorMode':'random'
              },
              'time_begin':'2011-06-28T23:43:00-06:00',
              'polygons':[[
                 [-106.4,35.0],
                 [-106.4,35.9],
                 [-106.0,35.9],
                 [-106.4,35.0]
                ],[
                 [-105.4,35.0],
                 [-105.4,35.9],
                 [-105.0,35.9],
                 [-105.4,35.0]
                ]
              ]
              },
          ]
    }
    contour2kml(data,'selftest.kml')
