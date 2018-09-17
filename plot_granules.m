function plot_granules(granules)
% Call:
% plot_granules(granules)
%
% Description:
% Plot all the granules in a matlab strucure. The matlab structure has to
% contain a first layer of satellite products, a second layer of dates and
% finally the lon, lat and fire information in the third layer.
%
% Developed in Matlab 9.2.0.556344 (R2017a) on MACINTOSH. 
% Angel Farguell (angel.farguell@gmail.com), 2018-08-24
%-------------------------------------------------------------------------

granul=fields(granules);
for ii=1:length(granul)
    lon=granules.(granul{ii}).lon;
    lat=granules.(granul{ii}).lat;
    fire=granules.(granul{ii}).fire;
    tit=strcat({'Plot of the '},granul{ii},{' granule.'});
    figure, h=pcolor(lon,lat,fire); title(tit,'Interpreter','none'); set(h,'EdgeColor','None'), cmfire, drawnow
end

end