import imageio, xmltodict, json
import numpy as np

with open('doc.kml','r') as f:
    s=f.read()

d=xmltodict.parse(s)
#print (json.dumps(d, indent=4, separators=(',', ': ')))

g = d["kml"]["Document"]["Folder"]["GroundOverlay"]

j = {}
n = 0
for i in g:
    n += 1
    file_s = i["name"] + '.png'
    time_s = i["gx:TimeStamp"]["when"]
    bbox_s = i["gx:LatLonQuad"]["coordinates"].replace('\t','')
    corners = map(float,bbox_s.replace('\n',',').split(','))
    a = {"file":file_s,"time":time_s,"corners":corners}
    j.update({n:a})

c = "doc.json"
with open(c,"w") as f:
    json.dump(j, f, indent=4, separators=(',', ': '))

print ("created file "+  c)
