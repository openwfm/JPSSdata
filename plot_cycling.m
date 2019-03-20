function plot_cycling(file,varargin)
% Call:
% plot_cycling(file) 
% plot_cycling(file,bounds)
%
% Description:
% Plots a granule from make_mat.py, input file for cycling
%
% Inputs:
%   file    matlab file output of make_mat.py
%   bounds  optional: array of [min_lon, max_lon, min_lat, max_lat]
%
% Developed in Matlab 9.2.0.556344 (R2017a) on MACINTOSH. 
% Angel Farguell (angel.farguell@gmail.com), 2018-08-24
%-------------------------------------------------------------------------


load(file);

[Nlat,Nlon] = size(data);
xx = linspace(geotransform(1),geotransform(1)+geotransform(2)*(Nlon-1),Nlon);
yy = linspace(geotransform(4)-geotransform(6)*(Nlat-1),geotransform(4),Nlat);
[lon,lat] = meshgrid(xx,yy);

figure, h=pcolor(lon,lat,data); cmfire()
set(h,'EdgeColor','None'); 

if nargin > 1
    bounds = varargin{1};
    mask = logical((lon > bounds(1)).*(lon < bounds(2)).*(lat > bounds(3)).*(lat < bounds(4)));
    figure, scatter(lon(mask),lat(mask),100,data(mask),'filled','s'); cmfire()
    xlim([min(lon(mask)),max(lon(mask))]);
    ylim([min(lat(mask)),max(lat(mask))])
end

end