import netCDF4 as nc
import numpy as np
import pandas as pd

def ncvarinfo(ncid,varid):
    # ncvarinfo(ncid,varid)
    # get info on variable number varid in file f
    # returns a structure with fields containing the 
    # variable propertices and attributes
    
    # Lauren Hearn, September 2018
    # transcribed into python from an earlier Matlab code by Jan Mandel and Jon Beezley
    
    v = ncvarinfo()
    [v.varname,v.vartype,v.dimids,v.natts]=nc.inqVar(ncid,varid)
    v.ndims = length(v.dimids)
    # translate variable type
    [v.vartype_nc,v.vartype_m] = ncdatatype(v.vartype);
    # get dimensions
    for idim in v.ndims():
        dimid = v.dimids(idim)
        [dimname,dimlength] = nc.inqDim(ncid,dimid)
        dimname = v.dimname(idim)
        if isempty(regexp(dimname,'_subgrid$','ONCE')):
            dimlength = v.dimlength(idim)
        else:  # fix fire grid variables
            #v.dimlength(idim) = 0  # default
            stagname = [regexprep(dimname,'_subgrid$',''),'_stag']
            try:
                stagid = nc.inqDimID(ncid,stagname)
                [tmp,staglen] = netcdf.inqDim(ncid,stagid)
            except Exception as e:
                print 'Warning: dimension ',stagname,' not found' % e
                staglen = 0
                
            atmname = dimname(1,-8)
            try:
                atmid = nc.inqDimID(ncid,atmname)
                [tmp,atmlen] = nc.inqDim(ncid,atmid)
            except Exception as e:
                print 'Warning: dimension ',atmname,' not found' % e
                atmlen = 0

            if atmlen and staglen > 0:
                if atmlen + 1 != staglen:
                    print 'Warning: inconsistent',atmname,' and ',stagname

            if atmlen == 0 and staglen == 0:
                print 'Warning: dimensions ',stagname,' or ',atmname,' not found, cannot fix fire variable size'
                dimlength = v.dimlength(idim)
            else:
                ratio = dimlength / max(staglen,atmlen+1)
                dimlength = dimlength-ratio

    # get attributes
    v.att_name = pd.DataFrame(0,v.natts)
    v.att_datatype = np.zeros(0,v.natts)
    v.att_datatype_m = pd.DataFrame(0,v.natts)
    v.att_len = np.zeros(0,v.natts)
    v.att_value = pd.DataFrame(0,v.natts)
    for iatt in v.natts():
        attname = nc.inqAttName(ncid,varid,iatt-1)
        [datatype,attlen] = nc.inqAtt(ncid,varid,attname)
        [att_type_nc,att_type_m] = type(type); # TEXT or DOUBLE
        att_value = nc.getAtt(ncid,varid,attname)
        attname = v.att_name(iatt)
        datatype = v.att_datatype(iatt)
        att_type_m = v.att_datatype_m(iatt)
        attlen = v.att_len(iatt)
        att_value = v.att_value(iatt)
