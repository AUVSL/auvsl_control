function [y,stop] = fcn(u1,u2)

persistent pathcount;
persistent pathlength;
persistent path; %error in code, initially used path but this is already a matlab function
persistent pathPre;

if isempty(pathcount)
   pathcount=1;     
   pathPre=u2;
   [pathlength,~]=size(pathPre);  
   path=[pathPre; pathPre(pathlength,1)+100 pathPre(pathlength,2)+100]; %sim takes multiple iterations to stop after reaching final point, so this prevents it from breaking
end

pos=[u1(5) u1(6)]; % [x,y] 
currentPoint=path(pathcount,:);
target=path(pathcount+1,:);

A = [(currentPoint(2)-target(2)), target(1)-currentPoint(1); target(1)-currentPoint(1), target(2)-currentPoint(2)];
b = [ target(1)*currentPoint(2)-currentPoint(1)*target(2); pos(1)*(target(1)-currentPoint(1)) + pos(2)*(target(2)-currentPoint(2))];
proj = (A\b)'; %projected point on the line between  .
projLen=dot(proj-currentPoint,target-currentPoint)/norm(target-currentPoint)^2; %the distance along the projected line, where 0 is the start point and 1 is the end point

if ((projLen>1))
    pathcount=pathcount+1;
end

%check if at destination and terminate sim if there
pathflag=0;
if (pathcount==pathlength)
    
    pathflag=1;
end

if ((pathcount==(pathlength-1))||(pathcount==(pathlength)))
    a=path(pathcount,:);
    b=path(pathcount+1,:);
    post=path(pathcount+1,:);
else
    a=path(pathcount,:);
    b=path(pathcount+1,:);
    post=path(pathcount+2,:);
end

y = [a(1);a(2);b(1);b(2);post(1);post(2)];
stop=pathflag;