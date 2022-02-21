% Init %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear, clc;

N = 12;
use_past_points = false;
save_param = true;

if use_past_points == true
    fprintf("Parameters loaded.\n")
    load("./perspective_images/param.mat")
end

% Load imgs %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

im1 = imread(".\perspective_images\1.jpg");
im2 = imread(".\perspective_images\2.jpg");

if use_past_points == false
    figure(1);
    imshow(im1)
    title("1. From (points selected : 0/"+num2str(N)+")")
    figure(2);
    imshow(im2)
    title("2. To (points selected : 0/"+num2str(N)+")")
end

% Select match points %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if use_past_points == false
    % Select points in "from" img
    figure(1);
    hold on
    points1 = zeros([N,2]);
    for i = 1:N
        points1(i,:) = ginput(1);
        scatter(points1(i,1), points1(i,2), 40, ...
                  'MarkerEdgeColor',[0 .5 .5],...
                  'MarkerFaceColor',[0 .7 .7],...
                  'LineWidth',1.5)
        text(points1(i,1), points1(i,2), num2str(i), ...
            "HorizontalAlignment","center", ...
            "VerticalAlignment","middle")
        title("1. From (points selected : "+num2str(i)+"/"+num2str(N)+")")
    end
    hold off
    pause(0.5)
    
    % Select points in "To" img
    figure(2);
    hold on
    points2 = zeros([N,2]);
    for i = 1:N
        points2(i,:) = ginput(1);
        scatter(points2(i,1), points2(i,2), 40, ...
                  'MarkerEdgeColor',[0 .5 .5],...
                  'MarkerFaceColor',[0 .7 .7],...
                  'LineWidth',1.5)
        text(points2(i,1), points2(i,2), num2str(i), ...
            "HorizontalAlignment","center", ...
            "VerticalAlignment","middle")
        title("2. To (points selected : "+num2str(i)+"/"+num2str(N)+")")
    end
    hold off
    pause(0.5)
end

if save_param == true
    fprintf("Points saved.\n")
    save("./perspective_images/param.mat", "points1", "points2")
end

% Cal trans matrix and epipoles %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

F1 = stereo_vision_mat_cal(points2, points1);   % p2·F1·p1 = 0
F2 = stereo_vision_mat_cal(points1, points2);   % p1·F2·p2 = 0

% Transform %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

figure(3);
sgtitle("3. Result (choose points for epipolar lines)")
ax1 = subplot(1,2,1);
imshow(im1)
ax2 = subplot(1,2,2);
imshow(im2)
while true
    point = ginput(1)
    point = [point'; 1];
    [~, im1_click] = ismember(gca, ax1);
    [~, im2_click] = ismember(gca, ax2);
    
    if im1_click == 1       % Click on im1
        subplot(1,2,2)
        hold on
        temp = F1 * point;
        a = -temp(1)/temp(2);   % slope
        b = -temp(3)/temp(2);   % bias
        plot([-1e2, 1e6], [-1e2*a+b, 1e6*a+b], 'color', '#0072BD')
        hold off
    elseif im2_click == 1   % Click on im2
        subplot(1,2,1)
        hold on
        temp = F2 * point;
        a = -temp(1)/temp(2);   % slope
        b = -temp(3)/temp(2);   % bias
        plot([-1e2, 1e6], [-1e2*a+b, 1e6*a+b], 'color', '#0072BD')
        hold off
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Funcs %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function F = stereo_vision_mat_cal(points1, points2)
    % This function finds F so that p1*F*p2=0 (p2 transforms to p1)
    N = size(points1, 1);
    A = zeros([N,9]);
    % h = zeros([9,1]);
    
    for i = 1:N
        x1 = points1(i,1);
        y1 = points1(i,2);
        x2 = points2(i,1);
        y2 = points2(i,2);
        A(i,:) = [x1*x2 x1*y2 x1 y1*x2 y1*y2 y1 x2 y2 1];
    end
    
    f = func_svd_min_eigvec(A);
    f = f / f(end);
    
    F = [
        f(1), f(2), f(3);
        f(4), f(5), f(6);
        f(7), f(8), f(9)
    ];
end
