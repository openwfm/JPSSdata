function comparison_cycling(files,comparison,bounds)
% Call:
% comparison_cycling(files,comparison,bounds) 
%
% Description:
% Plots a comparison of different results from make_mat.py
%
% Inputs:
%   files       cell of file names to compare (size available: from 1 to 3)
%   comparison  file with original pixel detections
%   bounds      array of [min_lon, max_lon, min_lat, max_lat]
%
% Developed in Matlab 9.2.0.556344 (R2017a) on MACINTOSH. 
% Angel Farguell (angel.farguell@gmail.com), 2019-03-21
%-------------------------------------------------------------------------

nfiles = length(files);
s_mark = 30;

figure,

load(comparison);
ax(1) = subplot(2,2,1);
scatter(lons,lats,s_mark,fires,'filled','o');
cmfire;
xlim([bounds(1),bounds(2)]);
ylim([bounds(3),bounds(4)]);
title('Original nodes to compare');

for k=2:nfiles+1 
    [~,name{k-1},~] = fileparts(files{k-1});
    load(files{k-1});
    [rows,cols] = size(data);
    geo = geotransform;
    Xpixel=(0:cols-1)+0.5;
    Ypixel=(0:rows-1)+0.5;
    xx = geo(1)+Xpixel*geo(2);
    yy = geo(4)+Ypixel*geo(6);
    [lon,lat] = meshgrid(xx,yy);
    mask = logical((lon > bounds(1)).*(lon < bounds(2)).*(lat > bounds(3)).*(lat < bounds(4)));
    
    ax(k) = subplot(2,2,k);
    scatter(lon(mask),lat(mask),s_mark,data(mask),'filled','o'); cmfire;
    xlim([bounds(1),bounds(2)]);
    ylim([bounds(3),bounds(4)]);
    title(sprintf('Fire mesh of %s',name{k-1}),'Interpreter','none');
end

linkaxes(ax,'xy');

end