function plot_cycling(file,varargin)
% Call:
% plot_cycling(file) 
% plot_cycling(file,bounds)
% plot_cycling(file,bounds,comparison)
%
% Description:
% Plots a granule from make_mat.py, input file for cycling
%
% Inputs:
%   file        matlab file output of make_mat.py
%   bounds      optional: array of [min_lon, max_lon, min_lat, max_lat]
%   comparison  file with original pixel detections
%
% Developed in Matlab 9.2.0.556344 (R2017a) on MACINTOSH. 
% Angel Farguell (angel.farguell@gmail.com), 2019-03-20
%-------------------------------------------------------------------------

[~,name,~] = fileparts(file);
s_mark = 50;

load(file);

[Nlat,Nlon] = size(data);
xx = linspace(geotransform(1),geotransform(1)+geotransform(2)*(Nlon-1),Nlon);
yy = linspace(geotransform(4)-geotransform(6)*(Nlat-1),geotransform(4),Nlat);
[lon,lat] = meshgrid(xx,yy);

figure, h=pcolor(lon,lat,data); 
cmfire;
set(h,'EdgeColor','None'); 
title(sprintf('Whole granule of %s',name),'Interpreter','none');

if nargin > 1
    bounds = varargin{1};
    mask = logical((lon > bounds(1)).*(lon < bounds(2)).*(lat > bounds(3)).*(lat < bounds(4)));
    figure, scatter(lon(mask),lat(mask),s_mark,data(mask),'filled','s'); 
    cmfire;
    xlim([bounds(1),bounds(2)]);
    ylim([bounds(3),bounds(4)]);
    title(sprintf('Fire mesh of %s',name),'Interpreter','none');
    if nargin > 2
        compare = varargin{2};
        load(compare)
        figure, 
        scatter(lons,lats,s_mark,fires,'filled','o');
        cmfire;
        xlim([bounds(1),bounds(2)]);
        ylim([bounds(3),bounds(4)]);
        title('Original nodes to compare');
    end
end

end