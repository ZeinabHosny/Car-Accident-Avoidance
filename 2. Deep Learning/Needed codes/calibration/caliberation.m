%% Correct Image for Lens Distortion
%% Create a set of calibration images.

% Copyright 2015 The MathWorks, Inc.


images = imageSet(fullfile(toolboxdir('vision'),'visiondata','calibration','fishEye'));  %%checker board from matlab images
%images = imageSet('C:\Users\SMSM-TECH\Desktop\m\Untitled Folder\');

%% Detect the calibration pattern.

%%%%matlab calibration%%%%%%%%%%%%
[imagePoints, boardSize] = detectCheckerboardPoints(images.ImageLocation);
%% Generate the world coordinates of the corners of the squares. The square size is in millimeters.
squareSize = 29; 
worldPoints = generateCheckerboardPoints(boardSize, squareSize);
%% Calibrate the camera.
cameraParams = estimateCameraParameters(imagePoints, worldPoints);
% Remove lens distortion and display the results.

% I = imread('real_Image1.jpg');
% J1 = undistortImage(I, cameraParams); 
% %% Display the original and corrected images.
% figure; imshowpair(I, J1, 'montage');
% title('Original Image (left) vs. Corrected Image (right)');
% 
% J2 = undistortImage(I, cameraParams, 'OutputView', 'full');
% figure; imshow(J2);
% title('Full Output View');
% 
% 
% 
%%%%%%%%%%%%%try

for a = 1:391
   filename = ['test_acc_real-' num2str(a,'%01d') '_1' '.jpg'];    %%num2str to numbering images
   img = imread(filename);
   J1 = undistortImage(img, cameraParams);
   %figure; imshowpair(img, J1, 'montage');
title('Original Image (left) vs. Corrected Image (right)');
   % do something with img
   savename=['cal_maha' num2str(a,'%01d') '_1' '.jpg'];
   imwrite(J1,savename)
end

% filename = ['real-' num2str(a,'%01d') '.jpg'];
% img = imread(filename);
% figure
% imshow(img)
% RGB2 = imresize(img, [720 720]);
% figure
% imshow(RGB2)
% imwrite(RGB2,'new.jpg')
% y=imcrop(RGB2,[480 720 0 0]);
% figure
% imshow(y)
