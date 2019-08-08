%     shuttleVideo = VideoReader('VID-20190108-WA0015.MP4');
%     ii = 1;
% 
%     while hasFrame(shuttleVideo)
%        img = readFrame(shuttleVideo);
%        filename = [sprintf('%03d',ii) '.jpg'];
%        fullname = fullfile('dodo','images',filename);
%        imwrite(img,fullname)    % Write out to a JPEG file (img1.jpg, img2.jpg, etc.)
%        ii = ii+1;
%     end
clear
clc
 vid=VideoReader('000594.mp4');
 numFrames = vid.NumberOfFrames;
 n=numFrames;
 for i = 1:5:n
 frames = read(vid,i);
 imwrite(frames,['f51_i' int2str(i), '.jpg']);
 %im(i)=image(frames);
 end