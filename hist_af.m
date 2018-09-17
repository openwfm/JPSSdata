function pixels=hist_af(af)
for i=1:10
    h(i)=sum(af(:)==i-1);
end
pixels.unknown= count(1)+count(2)+count(3)+count(7);
pixels.water  = count(4);
pixels.cloud  = count(5);
pixels.land   = count(6);
pixels.fire_low=count(8);
pixels.fire_med=count(9);
pixels.fire_high=count(10);
pixels.total = prod(size(af));


