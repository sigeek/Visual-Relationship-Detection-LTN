clc
clear all
close all
addpath('evaluation');

if ~exist('results/demo', 'file')
    mkdir('results/demo');
end

if ~exist('results/demo_zeroShot', 'file')
    mkdir('results/demo_zeroShot');
end

%load ground truth
load('evaluation/gt.mat', 'gt_obj_bboxes', 'gt_sub_bboxes', 'gt_tuple_label');

%load results
load('results/relationship_det_result.mat', 'rlp_labels_ours', 'rlp_confs_ours', 'sub_bboxes_ours', 'obj_bboxes_ours');
%% demo some good examples
load('samples/sampleMat.mat', 'sampleMat')
for uu = 1 : size(sampleMat,1)
    result_visualization( sampleMat(uu,1), sampleMat(uu,2), 'results/demo/', rlp_labels_ours, rlp_confs_ours, sub_bboxes_ours, obj_bboxes_ours);
end

%% demo with ground truth
load('samples/sampleMatGt.mat', 'sampleMat')
for uu = 1 : size(sampleMat,1)
    result_visualization_gt( sampleMat(uu,1), sampleMat(uu,2),  sampleMat(uu,3),'results/demo_gt/', ...
        rlp_labels_ours, rlp_confs_ours, sub_bboxes_ours, obj_bboxes_ours, ...
        gt_sub_bboxes, gt_obj_bboxes, gt_tuple_label);
end

%% demo some good zero-shot examples
load('samples/sampleMatZeroShot.mat', 'sampleMatZeroShot')
for uu = 1 : size(sampleMatZeroShot,1) 
    result_visualization(sampleMatZeroShot(uu,1), sampleMatZeroShot(uu,2), 'results/demo_zeroShot/', rlp_labels_ours, rlp_confs_ours, sub_bboxes_ours, obj_bboxes_ours);
end

%% demo some good zero-shot with ground truth
load('samples/sampleMatZeroShotGt.mat', 'sampleMatZeroShot')
for uu = 1 : size(sampleMatZeroShot,1) 
    result_visualization_gt(sampleMatZeroShot(uu,1), sampleMatZeroShot(uu,2), sampleMatZeroShot(uu,3),...
        'results/demo_zeroShot_gt/', ...
        rlp_labels_ours, rlp_confs_ours, sub_bboxes_ours, obj_bboxes_ours, ...
        gt_sub_bboxes, gt_obj_bboxes, gt_tuple_label);end
 
