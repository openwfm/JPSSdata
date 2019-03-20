function plot_cycling(file,varargin)
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