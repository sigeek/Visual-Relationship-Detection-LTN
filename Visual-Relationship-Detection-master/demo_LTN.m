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

% load results
load('results_LTN/relationship_det_result_KB_wc_2500.mat', 'rlp_labels_ours', 'rlp_confs_ours', 'sub_bboxes_ours', 'obj_bboxes_ours');
%% demo some good examples
load('samples/sampleMat.mat', 'sampleMat')
for uu = 1 : size(sampleMat,1)
    result_visualization_LTN( sampleMat(uu,1), sampleMat(uu,2), 'results_LTN/demo/', ...
        rlp_labels_ours, rlp_confs_ours, sub_bboxes_ours, obj_bboxes_ours, ...
        gt_sub_bboxes, gt_obj_bboxes, gt_tuple_label);
end

%% demo some good zero-shot examples
load('samples/sampleMatZeroShot.mat', 'sampleMatZeroShot')
for uu = 1 : size(sampleMatZeroShot,1) 
    result_visualization_LTN(sampleMatZeroShot(uu,1), sampleMatZeroShot(uu,2), 'results_LTN/demo_zeroShot/',...
    rlp_labels_ours, rlp_confs_ours, sub_bboxes_ours, obj_bboxes_ours,...
    gt_sub_bboxes, gt_obj_bboxes, gt_tuple_label);
end

 
