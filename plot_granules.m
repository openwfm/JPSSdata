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

prods=fields(granules);
for ii=1:length(prods)
    dates=fields(granules.(prods{ii}));
    for jj=1:length(dates)
        lon=granules.(prods{ii}).(dates{jj}).lon;
        lat=granules.(prods{ii}).(dates{jj}).lat;
        fire=granules.(prods{ii}).(dates{jj}).fire;
        tit=strcat({'Plot of the '},{prods{ii}},{' granule in date '},{dates{jj}});
        figure, h=pcolor(lon,lat,fire); title(tit); set(h,'EdgeColor','None'), cmfire
    end
end

end