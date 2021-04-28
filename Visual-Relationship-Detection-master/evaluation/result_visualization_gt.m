function result_visualization_gt(id, idx, idx_gt, saveFile, ...,
    rlp_labels_ours, rlp_confs_ours, sub_bboxes_ours, obj_bboxes_ours,...
    gt_sub_bboxes, gt_obj_bboxes, gt_tuple_label)

    dataset_test = 'samples'; 
    
    load('data/imagePath.mat')    
    load('data/relationListN.mat')
    load('data/objectListN.mat')
    if ~exist([dataset_test,'/',imagePath{id}])
        disp('please download scene graph dataset form')
        disp('############')
        return;
    end
    
    im = im2double(imread([dataset_test,'/',imagePath{id}]));
    vw = 4; 
    
    %definition of bounding boxes and labels (ground truth)
    box_sub_gt = gt_sub_bboxes{id}(idx_gt,:);
    box_obj_gt = gt_obj_bboxes{id}(idx_gt,:);

    str_sub_gt = char(objectListN(gt_tuple_label{id}(idx_gt,1)));
    str_obj_gt = char(objectListN(gt_tuple_label{id}(idx_gt,3)));
    str_rel_gt = char(relationListN(gt_tuple_label{id}(idx_gt,2)));
    
    % definition of bounding boxes and labels (predicted)
    box_sub_pred = sub_bboxes_ours{id}(idx,:);
    box_obj_pred = obj_bboxes_ours{id}(idx,:);

    str_sub_pred = char(objectListN(rlp_labels_ours{id}(idx,1)));
    str_obj_pred = char(objectListN(rlp_labels_ours{id}(idx,3)));
    str_rel_pred = char(relationListN(rlp_labels_ours{id}(idx,2)));


    %masks ground truth
    %subject mask
    mask = zeros(size(im));
    masks = zeros(size(im));
    mask(box_sub_gt(2):box_sub_gt(4),box_sub_gt(1):box_sub_gt(3),:) = 1;
    masks((box_sub_gt(2)+vw):(box_sub_gt(4)-vw),(box_sub_gt(1)+vw):(box_sub_gt(3)-vw),:) = 1; 
    mask = (mask - masks);
    mask(:,:,2:3) = -10*mask(:,:,2:3); 
    im = min(max(im + mask,0),1); 

    %object mask
    mask = zeros(size(im));
    masks = zeros(size(im));
    mask(box_obj_gt(2):box_obj_gt(4),box_obj_gt(1):box_obj_gt(3),:) = 1;
    masks((box_obj_gt(2)+vw):(box_obj_gt(4)-vw),(box_obj_gt(1)+vw):(box_obj_gt(3)-vw),:) = 1; 
    mask = (mask - masks);
    mask(:,:,2:3) = -10*mask(:,:,1:2); 
    im = min(max(im + mask,0),1); 

    sub_gt.cx = round((box_sub_gt(2) + box_sub_gt(4))/2);
    sub_gt.cy = round((box_sub_gt(1) + box_sub_gt(3))/2);

    obj_gt.cx = round((box_obj_gt(2) + box_obj_gt(4))/2);
    obj_gt.cy = round((box_obj_gt(1) + box_obj_gt(3))/2);

    rel_gt.cx = round((sub_gt.cx + obj_gt.cx)/2);
    rel_gt.cy = round((sub_gt.cy + obj_gt.cy)/2);
    
    %masks predictions
    %subject mask
    mask = zeros(size(im));
    masks = zeros(size(im));
    mask(box_sub_pred(2):box_sub_pred(4),box_sub_pred(1):box_sub_pred(3),:) = 1;
    masks((box_sub_pred(2)+vw):(box_sub_pred(4)-vw),(box_sub_pred(1)+vw):(box_sub_pred(3)-vw),:) = 1; 
    mask = (mask - masks);
    mask(:,:,1:2) = -10*mask(:,:,2:3);
    im = min(max(im + mask,0),1); 

    %object mask
    mask = zeros(size(im));
    masks = zeros(size(im));
    mask(box_obj_pred(2):box_obj_pred(4),box_obj_pred(1):box_obj_pred(3),:) = 1;
    masks((box_obj_pred(2)+vw):(box_obj_pred(4)-vw),(box_obj_pred(1)+vw):(box_obj_pred(3)-vw),:) = 1; 
    mask = (mask - masks);
    mask(:,:,1:2) = -10*mask(:,:,1:2);
    im = min(max(im + mask,0),1); 

    sub_pred.cx = round((box_sub_pred(2) + box_sub_pred(4))/2);
    sub_pred.cy = round((box_sub_pred(1) + box_sub_pred(3))/2);

    obj_pred.cx = round((box_obj_pred(2) + box_obj_pred(4))/2);
    obj_pred.cy = round((box_obj_pred(1) + box_obj_pred(3))/2);

    rel_pred.cx = round((sub_pred.cx + obj_pred.cx)/2);
    rel_pred.cy = round((sub_pred.cy + obj_pred.cy)/2);
    
   
    gcf=figure;  imshow(im);hold on
    
    %ground truth
    text(sub_gt.cy, sub_gt.cx, [str_sub_pred, ' (gt)'] ,'color','red','fontsize',20);hold on
    text(obj_gt.cy, obj_gt.cx, [str_obj_pred, ' (gt)'] ,'color','red','fontsize',20); hold on
    if size(im,1) > size(im,2)
        strRep = [ '<',str_sub_pred , ', ' ,str_rel_pred ,', ',  str_obj_pred , ...
            '> score: ', sprintf('%0.1f \n',rlp_confs_ours{id}(idx)), ...
            '(<',str_sub_gt , ', ' ,str_rel_gt ,', ',  str_obj_gt , '>)'];
        text(1, round(size(im,2)/10), strRep,'color','green','fontsize',16);hold on
    else
        strRep = [ '<',str_sub_pred , ', ' ,str_rel_pred ,', ',  str_obj_pred , ...
            '> score: ', sprintf('%0.1f \n',rlp_confs_ours{id}(idx)), ...
            '(<',str_sub_gt , ', ' ,str_rel_gt ,', ',  str_obj_gt , '>)'];
        text(round(size(im,1)/7), round(size(im,2)/10), strRep,'color','green','fontsize',16);hold on
    end
    
    
    
    
    saveas(gcf,[saveFile, num2str(id),'_', num2str(idx)],'png');
    close all;
  
end
