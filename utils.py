# Copyright (C) 2013-2016 Martin Vejmelka, UC Denver

import json,sys

class Dict(dict):
    """
    A dictionary that allows member access to its keys.
    A convenience class.
    """

    def __init__(self, d):
        """
        Updates itself with d.
        """
        self.update(d)

    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError:
            raise AttributeError(item)

    def __setattr__(self, item, value):
        self[item] = value

def load_cfg():
    # load the system configuration
    cfg = Dict([])
    try:
        f_cfg = Dict(json.load(open('conf.json')))
    except IOError:
        print 'Warning: any conf.json file specified, creating defaults...'
        f_cfg = Dict([])

    # Set default method settings
    try:
        # minimum confidence level for the satellite pixels to be considered
        cfg.minconf = f_cfg['method_settings'].get('minconf',70)
        # use 3D space-time points (or bounds)
        cfg.cloud = f_cfg['method_settings'].get('cloud',True)
        # dynamic penalization term?
        cfg.dyn_pen = f_cfg['method_settings'].get('dyn_pen',False)
        # if not so, 5-fold cross validation for C and gamma?
        if cfg.dyn_pen:
            cfg.search = False
        else:
            cfg.search = f_cfg['method_settings'].get('search',False)
        # interpolation of the results into fire mesh (if apply to spinup case)
        cfg.fire_interp = f_cfg['method_settings'].get('fire_interp',False)
    except:
        cfg.minconf = 70
        cfg.cloud = True
        cfg.dyn_pen = False
        cfg.search = False
        cfg.fire_interp = False

    # Set default data paths
    try:
        # if ignitions are known: ([lons],[lats],[dates]) where lons and lats in degrees and dates in ESMF format
        # examples: "igns" : "([100],[45],['2015-05-15T20:09:00'])" or "igns" : "([100,105],[45,39],['2015-05-15T20:09:00','2015-05-15T23:09:00'])"
        cfg.igns = eval(f_cfg['data_paths'].get('igns','None'))
        # if infrared perimeters: path to KML files
        # examples: perim_path = './pioneer/perim'
        cfg.perim_path = f_cfg['data_paths'].get('perim_path','')
        # if forecast wrfout: path to netcdf wrfout forecast file
        # example: forecast_path = './patch/wrfout_patch'
        cfg.forecast_path = f_cfg['data_paths'].get('forecast_path','')
    except:
        cfg.igns = None
        cfg.perim_path = ''
        cfg.forecast_path = ''

    # Set default plot settings
    try:
        # plot observed information (googlearth.kmz with png files)
        cfg.plot_observed = f_cfg['plot_settings'].get('plot_observed',False)
        # if so, only fire detections?
        if cfg.plot_observed:
            cfg.only_fire = f_cfg['plot_settings'].get('only_fire',False)
        else:
            cfg.only_fire = False
    except:
        cfg.plot_observed = False
        cfg.only_fire = False

    # Set SVM default settings
    cfg.svm_settings = Dict([])
    # Plot settings
    try:
        # plot original data
        cfg.svm_settings.plot_data = f_cfg['plot_settings']['plot_svm'].get('plot_data',False)
        # plot scaled data with artificial data
        cfg.svm_settings.plot_scaled = f_cfg['plot_settings']['plot_svm'].get('plot_scaled',False)
        # plot decision volume
        cfg.svm_settings.plot_decision = f_cfg['plot_settings']['plot_svm'].get('plot_decision',False)
        # plot polynomial approximation
        cfg.svm_settings.plot_poly = f_cfg['plot_settings']['plot_svm'].get('plot_poly',False)
        # plot full hyperplane vs detections with support vectors
        cfg.svm_settings.plot_supports = f_cfg['plot_settings']['plot_svm'].get('plot_supports',False)
        # plot resulting fire arrival time vs detections
        cfg.svm_settings.plot_result = f_cfg['plot_settings']['plot_svm'].get('plot_result',False)
    except:
        cfg.svm_settings.plot_data = False
        cfg.svm_settings.plot_scaled = False
        cfg.svm_settings.plot_decision = False
        cfg.svm_settings.plot_poly = False
        cfg.svm_settings.plot_supports = False
        cfg.svm_settings.plot_result = False
    # Method settings
    try:
        # normalize the input data between 0 and 1
        cfg.svm_settings.norm = f_cfg['method_settings']['svm_settings'].get('norm',True)
        # if not Nans in the data are wanted (all Nans are going to be replaced by the maximum value)
        cfg.svm_settings.notnan = f_cfg['method_settings']['svm_settings'].get('notnan',True)
        # artificial creation of clear ground detections under real (preprocessing)
        cfg.svm_settings.artil = f_cfg['method_settings']['svm_settings'].get('artil',False)
        # if so, normalized vertical resolution (from 0 to 1) of the aggregation
        if cfg.svm_settings.artil:
            cfg.svm_settings.hartil = f_cfg['method_settings']['svm_settings'].get('hartil',.2)
        else:
            cfg.svm_settings.hartil = 0
        # artificial creation of fire detections above real (preprocessing)
        cfg.svm_settings.artiu = f_cfg['method_settings']['svm_settings'].get('artiu',True)
        # if so, normalized vertical resolution (from 0 to 1) of the aggregation
        if cfg.svm_settings.artiu:
            cfg.svm_settings.hartiu = f_cfg['method_settings']['svm_settings'].get('hartiu',.1)
        else:
            cfg.svm_settings.hartiu = 0
        # creation of an artificial mesh of clear ground detections at the bottom
        cfg.svm_settings.downarti = f_cfg['method_settings']['svm_settings'].get('downarti',True)
        # if so, how to the bottom normalized between 0 and 1 and confidence level of the artificial detections
        if cfg.svm_settings.downarti:
            cfg.svm_settings.dminz = f_cfg['method_settings']['svm_settings'].get('dminz',.1)
            cfg.svm_settings.confal = f_cfg['method_settings']['svm_settings'].get('confal',100)
        else:
            cfg.svm_settings.dminz = 0
            cfg.svm_settings.confal = 0
        # creation of an artificial mesh of fire detections at the top
        cfg.svm_settings.toparti = f_cfg['method_settings']['svm_settings'].get('toparti',False)
        # if so, how to the top normalized between 0 and 1 and confidence level of the artificial detections
        if cfg.svm_settings.toparti:
            cfg.svm_settings.dmaxz = f_cfg['method_settings']['svm_settings'].get('dmaxz',.1)
            cfg.svm_settings.confau = f_cfg['method_settings']['svm_settings'].get('confau',100)
        else:
            cfg.svm_settings.dmaxz = 0
            cfg.svm_settings.confau = 0
    except:
        cfg.svm_settings.notnan = True
        cfg.svm_settings.artil = False
        cfg.svm_settings.hartil = 0
        cfg.svm_settings.artiu = True
        cfg.svm_settings.hartiu = .1
        cfg.svm_settings.downarti = True
        cfg.svm_settings.dminz = .1
        cfg.svm_settings.confal = 100
        cfg.svm_settings.toparti = False
        cfg.svm_settings.dmaxz = 0
        cfg.svm_settings.confau = 0
    cfg.svm_settings.search = cfg.search

    # Set AppKey for NRT downloads from https://nrt3.modaps.eosdis.nasa.gov/profile/app-keys
    try:
        cfg.appkey = f_cfg.get('appkey',None)
    except:
        cfg.appkey = None

    return cfg
