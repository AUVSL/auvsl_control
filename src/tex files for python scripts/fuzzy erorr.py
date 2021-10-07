function y  = fcn(u)

%jackal dyanamic model
pos=[u(5) u(6)]; % [x,y]
q=[u(1) u(2) u(3) u(4)]; % orientation quaternion
currentAngle=atan2(2*(q(1)*q(4)+q(2)*q(3)),1-2*(q(3)^2+q(4)^2));% Angle robot is currently facing

currentPoint=[u(7) u(8)];
target=[u(9) u(10)];
future=[u(11) u(12)];

A = [(currentPoint(2)-target(2)), target(1)-currentPoint(1); target(1)-currentPoint(1), target(2)-currentPoint(2)];
b = [ target(1)*currentPoint(2)-currentPoint(1)*target(2); pos(1)*(target(1)-currentPoint(1)) + pos(2)*(target(2)-currentPoint(2))];
proj = (A\b)'; %projected point on the line between  .

d = ( pos(1)-currentPoint(1) )*(target(2)-currentPoint(2)) - (pos(2)-currentPoint(2))*(target(1)-currentPoint(1));

if ( d >0)
    side = 1;
elseif ( d < 0)
    side = -1;
else
    side = 0;
end

distanceLine=norm(pos-proj)*side;
distanceTarget=sqrt((target(1)-pos(1))^2+(target(2)-pos(2))^2);

%wrap to +/-pi
if((currentAngle < -pi) || (pi < currentAngle))
    currentAngle = mod(currentAngle+pi, 2*pi)-pi;
end

farTarget=[.9*proj(1)+.1*target(1) .9*proj(2)+.1*target(2)];

th1 = atan2(farTarget(2)-pos(2), farTarget(1)-pos(1));
th2 = atan2(target(2)-currentPoint(2), target(1)-currentPoint(1));
th3 = atan2(future(2)-target(2), future(1)-target(1)); 

ThetaFar=th1-currentAngle;
ThetaNear=th2-currentAngle;
ThetaLookahead=th3-currentAngle;

%wrap to +/-pi
if((ThetaFar < -pi) || (pi < ThetaFar))
    ThetaFar = mod(ThetaFar+pi, 2*pi)-pi;
end

%wrap to +/-pi
if((ThetaNear < -pi) || (pi < ThetaNear))
    ThetaNear = mod(ThetaNear+pi, 2*pi)-pi;
end

%wrap to +/-pi
if((ThetaLookahead < -pi) || (pi < ThetaLookahead))
    ThetaLookahead = mod(ThetaLookahead+pi, 2*pi)-pi;
end

%y = [distanceLine;ThetaFar;ThetaNear;ThetaLookahead;distanceTarget]; 
y = [distanceLine;ThetaFar;ThetaNear]; 