clc,clear;
srcDir=uigetdir('Pre_test');
cd(srcDir);
allnames=struct2cell(dir('*.jpg'));
[k,len]=size(allnames);
for ii=1:len
    name=allnames{1,ii};
    I=imread(name);
% I = imread('h02060.jpg');
I=im2gray(I);%图像灰度化
figure;imshow(I)
title('原始图像','FontSize',14);
filterSize = 1;
sigma = 2;
denoisedImg = imgaussfilt(I, sigma, 'FilterSize', filterSize);
imshow(denoisedImg); title('高斯滤波去噪');
filterSize = 5;
smoothedImg = medfilt2(denoisedImg, [filterSize filterSize]);
imshow(smoothedImg);
title('中值平滑处理');
a = 1.6; % 缩放系数
b = -15; % 偏移量
new_img = uint8(a * double(smoothedImg) + b);
imshow(new_img);title('增强处理')
%阈值二值化法
% tz = imcomplement(new_img);
% BW = imbinarize(tz);%二值
% TT=~BW;
% figure;imshow(TT)
% 使用Otsu's方法自动计算阈值并二值化
threshold = graythresh(new_img);
binaryImage = imbinarize(new_img, threshold);

% 显示二值化图像
imshow(binaryImage);
title('Otsus Binary Image');
title('预处理结果','FontSize',14);
%特征提取
points = detectBRISKFeatures(binaryImage);
figure
imshow(I); hold on; plot(points,'ShowScale',false)
%确定文件储存路径
targetFolder='C:\Users\DELL\Desktop\新建文件夹 (2)\chucun';
   [~, nameNoExt, ext] = fileparts(name);
    fullFilePath = fullfile(targetFolder, [nameNoExt '_binary' ext]);
    
    % 保存二值化图像到指定文件夹
    imwrite(binaryImage, fullFilePath);
%为预处理后的图片重命名保存
% images{ii}=binaryImage;
% img=images{ii};
% fileName=['image' num2str(ii) '.jpg'];
% fullFilePath=fullfile(targetFolder,fileName);
% imwrite(img,fullFilePath);
end

