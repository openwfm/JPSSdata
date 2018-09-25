load result.mat

figure, mesh(fxlon,fxlat,U), title('U')
figure, mesh(fxlon,fxlat,L), title('L')
figure, mesh(fxlon,fxlat,T), title('T')

Ln=L; Ln(T==(time_scale_num(2)-time_scale_num(1)))=0;
figure, mesh(fxlon,fxlat,Ln), hold on, mesh(fxlon,fxlat,U), title('U vs L masked in T')
