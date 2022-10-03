close all
 %get_GradCAMviz(cls_dir,imgs_arra,class)
%cls_dir = ".\train_\Latino_Hispanic\";
%cls_dir = ".\train_\Southeast Asian\";
cls_dir = ".\train_\East Asian\";

hispanic= ["259.jpg" "2324.jpg" "402.jpg" "3605.jpg" "1184.jpg" "2289.jpg" "2542.jpg" "4961.jpg"];

mid_east = ["713.jpg" "78.jpg" "1102.jpg" "1483.jpg" "537.jpg" "1536.jpg" "2722.jpg" "3043.jpg" "2767.jpg" "3411.jpg" "4056.jpg" "4870.jpg" "5092.jpg" ];

black = ["433.jpg" "1212.jpg" "3423.jpg" "4709.jpg" "6145.jpg" "7390.jpg" "13233.jpg"  ];

low_qual =["678.jpg" "3033.jpg" "2288.jpg" "750.jpg" "4451.jpg" "5828.jpg" ];

black2 = ["43.jpg" "134.jpg" "1000.jpg"  "728.jpg" "2938.jpg" "3712.jpg" "3617.jpg" "4297.jpg"];

White2 = [ "4.jpg" "6.jpg" "44.jpg" "27.jpg" "376.jpg" "694.jpg" "1476.jpg" "1665.jpg" "2338.jpg" "3794.jpg"];

South_east_Asian =[496 219 496 996 906 1241 1583 1538 2068 2157 2808 3794 4069 4464 7972 450 5081 5191 7663 8413 8783 9288];
%South_east_Asian2 =[360 426 317 341 347 243  661 720 707 688 683 1265 1210 1859 2076 2182 2076 2851 3142 3660 3422 3689];
SEA_impt_ftrs =[ 496, 1241, 1583, 2068, 3794 ,4069 ,4464,360 ,1859, 661,3142];


%East_Asia = [681 1362 1061 1561 1436 1009 2382 2396 3032 4052 4122 1500 4329 5102 5222 6497 6688 6798];

East_Asia = [1362 1061 1561 1436 1009 1500 2382 2396 3032 4052 4122 4329 5102 5222 6497 6688 6798 4871 4496 4699 4775];
class_light_skin = [ 4631, 7061, 5477, 4399, 6435, 7038 ,7276, 8883, 9438, 2402,3691, 3752 ,5911 ,5915 ,3935 ,3962 ,5291 ,5480, 6297, 6483, 6435, 7038 ,7197 ,7250, 7415,8291];
class_dark_skin = [9112, 9487, 9488, 9306,9306, 9765, 10630, 11299, 11948, 2634, 3119, 2634, 3695, 7607, 5171, 7327, 9988, 2658, 11727,10001];

hisp_misclas = [class_light_skin,class_dark_skin];


%%
imgs_arra = East_Asia;

class_name  = "East Asian";

load('trained_alexnet.mat')

figure
hold on
for idx=1:1:length(hisp_misclas)
    for i =1:1:length(imgs_arra)
        %img_path = strcat(cls_dir,(imgs_arra(i)));
        img_path = strcat(cls_dir,string(imgs_arra(i)),".jpg");
        img = imread(img_path);
        inputSize = net.Layers(1).InputSize(1:2);
        img = imresize(img,inputSize);
        
        figure
        [YPred,scores] = classify(net,img);
        [~,topIdx] = maxk(scores, 3);
        topScores = scores(topIdx);
        topClasses = classes(topIdx);
        
        subplot(1,2,1), imshow(img)
        xlabel(strcat("actual:",class_name))
        titleString = compose("%s (%.3f)",topClasses,topScores');
        
        sgtitle(sprintf(join(titleString, "; ")));
         
        % figure
        %subplot(1,2,1), 
        %imshow(img)
        %title(strcat("original Image: ", class_name ))
        %imshow(img)
        hold on
    
        label = classify(net,img)
        scoreMap = gradCAM(net,img,label);
        %figure
        subplot(1,2,2),imshow(img)
        hold on
        imagesc(scoreMap,'AlphaData',0.5)
        colormap jet
        %title(strcat("pred: ",string(label)))
        
        xlabel(strcat("pred: ",string(label)))
        saveas(gcf,strcat(string(imgs_arra(i)),".jpg"));
    end 

end 