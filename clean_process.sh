#!/bin/bash

# binary files
rm data result perimeters decision 2> /dev/null
# matlab files
rm result.mat svm.mat 2> /dev/null
# kml files
rm fire_detections.kml nofire.kml perimeters_svm.kml 2> /dev/null
# kmz files
rm googlearth.kmz 2> /dev/null
# png files
rm original_data.png scaled_data.png decision.png support.png tign_g.png result.png 2> /dev/null
