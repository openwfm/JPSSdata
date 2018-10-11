load result.mat

tscale = 24*3600;

figure, mesh(fxlon,fxlat,U/tscale), title('Upper bound (U) from satellite fire detections'), xlabel('Longitude'), ylabel('Latitude'), zlabel('Time in days')
figure, mesh(fxlon,fxlat,L/tscale), title('Lower bound (L) from satellite fire detections'), xlabel('Longitude'), ylabel('Latitude'), zlabel('Time in days')
figure, mesh(fxlon,fxlat,T/tscale), title('Mask (T) surrounding the fire detections'), xlabel('Longitude'), ylabel('Latitude'), zlabel('Time in days')

Ln=L; Ln(T==(time_scale_num(2)-time_scale_num(1)))=0;
figure, mesh(fxlon,fxlat,Ln/tscale), hold on, mesh(fxlon,fxlat,U/tscale), title('Upper vs Lower bounds masked'), xlabel('Longitude'), ylabel('Latitude'), zlabel('Time in days')
