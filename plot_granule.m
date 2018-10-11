function plot_granule(granule)
% Call:
% plot_granule(granule)
%
% Description:
% Plot a granules in a matlab strucure. The matlab structure has to
% contain a first layer of satellite products, a second layer of dates and
% finally the lon, lat and fire information in the third layer.
%
% Developed in Matlab 9.2.0.556344 (R2017a) on MACINTOSH. 
% Angel Farguell (angel.farguell@gmail.com), 2018-08-24
%-------------------------------------------------------------------------


lon=granule.lon;
lat=granule.lat;
fire=granule.fire;
tit=strcat({'Plot of the granule '},granule.name);
figure, h=pcolor(lon,lat,fire); title(tit,'Interpreter','none'); set(h,'EdgeColor','None'), cmfire, drawnow

end