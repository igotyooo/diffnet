#! /bin/bash
DATA_OUT_DIR=/home/dgyoo/workspace/dataout/diffnet/UCF101_RGB_S1
ITERATION_VGGM='-batchSize 384 -epochSize 199 -numEpoch 30'
ITERATION_VGG16D='-batchSize 256 -epochSize 299 -numEpoch 30'
ITERATION_VGG16SD='-batchSize 192 -epochSize 398 -numEpoch 30'

#############################################################
### Image-level differentiation. ############################
### Epoch size is set to 4-times larger than the default. ###
# Temporal network.
CUDA_VISIBLE_DEVICES=0,1,2,3 th main.lua -task vi_vggmpt_d_slcls $ITERATION_VGGM -stride 4 -diffLevel 0 #(E28) 65.18
CUDA_VISIBLE_DEVICES=0,1,2,3 th main.lua -task vi_vggmpt_d_slcls $ITERATION_VGGM -stride 4 -diffLevel 0 -learnRate 1e-4,1e-4 -startFrom $DATA_OUT_DIR/vi_vggmpt_d_slcls,batchSize=384,diffLevel=0,epochSize=199,stride=4/model_028.t7 #(E25) 66.74
CUDA_VISIBLE_DEVICES=0,1 th testonly.lua -task vi_vggmpt_d_slcls $ITERATION_VGGM -stride 4 -diffLevel 0 -numGpu 2 -startFrom $DATA_OUT_DIR/vi_vggmpt_d_slcls,batchSize=384,diffLevel=0,epochSize=199,stride=4/model_028,batchSize=384,diffLevel=0,epochSize=199,learnRate=1e-4,1e-4,stride=4/model_020.t7 # 75.94
CUDA_VISIBLE_DEVICES=0,1,2,3 th main.lua -task vi_vgg16pt_d_slcls $ITERATION_VGG16D -stride 4 -diffLevel 0 #(E26) 69.17
CUDA_VISIBLE_DEVICES=0,1,2,3 th main.lua -task vi_vgg16pt_d_slcls $ITERATION_VGG16D -stride 4 -diffLevel 0 -learnRate 1e-4,1e-4 -startFrom $DATA_OUT_DIR/vi_vgg16pt_d_slcls,batchSize=256,diffLevel=0,epochSize=299,stride=4/model_026.t7 #(E28) 71.19
CUDA_VISIBLE_DEVICES=0,1 th testonly.lua -task vi_vgg16pt_d_slcls $ITERATION_VGG16D -stride 4 -diffLevel 0 -numGpu 2 -startFrom $DATA_OUT_DIR/vi_vgg16pt_d_slcls,batchSize=256,diffLevel=0,epochSize=299,stride=4/model_026,batchSize=256,diffLevel=0,epochSize=299,learnRate=1e-4,1e-4,stride=4/model_028.t7
# Momentum network.
CUDA_VISIBLE_DEVICES=0,1,2,3 th main.lua -task vi_vggmpt_sdsel_slcls $ITERATION_VGGM -stride 4 -branchAfter 0 -mergeAfter 5 -diffChance 0.9 #(E27) 68.30
CUDA_VISIBLE_DEVICES=0,1,2,3 th main.lua -task vi_vggmpt_sdsel_slcls $ITERATION_VGGM -stride 4 -branchAfter 0 -mergeAfter 5 -diffChance 0.9 -learnRate 1e-4,1e-4 -startFrom $DATA_OUT_DIR/vi_vggmpt_sdsel_slcls,batchSize=384,branchAfter=0,diffChance=0.9,epochSize=199,mergeAfter=5,stride=4/model_027.t7 #(E26) 69.28
CUDA_VISIBLE_DEVICES=0,1 th testonly.lua -task vi_vggmpt_sdsel_slcls $ITERATION_VGGM -stride 4 -branchAfter 0 -mergeAfter 5 -diffChance 0.9 -numGpu 2 -startFrom $DATA_OUT_DIR/vi_vggmpt_sdsel_slcls,batchSize=384,branchAfter=0,diffChance=0.9,epochSize=199,mergeAfter=5,stride=4/model_027,batchSize=384,branchAfter=0,diffChance=0.9,epochSize=199,learnRate=1e-4,1e-4,mergeAfter=5,stride=4/model_026.t7 # 76.79
CUDA_VISIBLE_DEVICES=4,5,6,7 th main.lua -task vi_vgg16pt_sdsel_slcls $ITERATION_VGG16SD -stride 4 -branchAfter 0 -mergeAfter 13 -diffChance 0.9 #(E21) 72.85
CUDA_VISIBLE_DEVICES=4,5,6,7 th main.lua -task vi_vgg16pt_sdsel_slcls $ITERATION_VGG16SD -stride 4 -branchAfter 0 -mergeAfter 13 -diffChance 0.9 -learnRate 1e-4,1e-4 -startFrom $DATA_OUT_DIR/vi_vgg16pt_sdsel_slcls,batchSize=192,branchAfter=0,diffChance=0.9,epochSize=398,stride=4/model_021.t7 #(E08) 75.23
CUDA_VISIBLE_DEVICES=4,5 th testonly.lua -task vi_vgg16pt_sdsel_slcls $ITERATION_VGG16SD -stride 4 -branchAfter 0 -mergeAfter 13 -diffChance 0.9 -numGpu 2 -startFrom $DATA_OUT_DIR/vi_vgg16pt_sdsel_slcls,batchSize=192,branchAfter=0,diffChance=0.9,epochSize=398,stride=4/model_021,batchSize=192,branchAfter=0,diffChance=0.9,epochSize=398,learnRate=1e-4,1e-4,stride=4/model_010.t7 # 82.42

#############################################################
### Temporal stride. ########################################
### Epoch size is set to 4-times larger than the default. ###
# Temporal network.
CUDA_VISIBLE_DEVICES=0,1,2,3 th main.lua -task vi_vggmpt_d_slcls $ITERATION_VGGM -stride 1 -diffLevel 1 #(E30) 62.53
CUDA_VISIBLE_DEVICES=0,1,2,3 th main.lua -task vi_vggmpt_d_slcls $ITERATION_VGGM -stride 1 -diffLevel 1 -learnRate 1e-4,1e-4 -startFrom $DATA_OUT_DIR/vi_vggmpt_d_slcls,batchSize=384,epochSize=199,stride=1/model_030.t7 #(E28) 64.86
CUDA_VISIBLE_DEVICES=0,1 th testonly.lua -task vi_vggmpt_d_slcls $ITERATION_VGGM -stride 1 -diffLevel 1 -numGpu 2 -startFrom $DATA_OUT_DIR/vi_vggmpt_d_slcls,batchSize=384,epochSize=199,stride=1/model_030,batchSize=384,epochSize=199,learnRate=1e-4,1e-4,stride=1/model_028.t7 # 75.79
CUDA_VISIBLE_DEVICES=0,1,2,3 th main.lua -task vi_vggmpt_d_slcls $ITERATION_VGGM -stride 2 -diffLevel 1 #(E23) 68.67
CUDA_VISIBLE_DEVICES=0,1,2,3 th main.lua -task vi_vggmpt_d_slcls $ITERATION_VGGM -stride 2 -diffLevel 1 -learnRate 1e-4,1e-4 -startFrom $DATA_OUT_DIR/vi_vggmpt_d_slcls,batchSize=384,epochSize=199/model_023.t7 #(E23) 69.73
CUDA_VISIBLE_DEVICES=0,1 th testonly.lua -task vi_vggmpt_d_slcls $ITERATION_VGGM -stride 2 -diffLevel 1 -numGpu 2 -startFrom $DATA_OUT_DIR/vi_vggmpt_d_slcls,batchSize=384,epochSize=199/model_023,batchSize=384,epochSize=199,learnRate=1e-4,1e-4/model_023.t7 # 78.66
CUDA_VISIBLE_DEVICES=0,1,2,3 th main.lua -task vi_vggmpt_d_slcls $ITERATION_VGGM -stride 8 -diffLevel 1 #(E26) 69.49
CUDA_VISIBLE_DEVICES=0,1,2,3 th main.lua -task vi_vggmpt_d_slcls $ITERATION_VGGM -stride 8 -diffLevel 1 -learnRate 1e-4,1e-4 -startFrom $DATA_OUT_DIR/vi_vggmpt_d_slcls,batchSize=384,epochSize=199,stride=8/model_026.t7 #(E19) 70.63
CUDA_VISIBLE_DEVICES=0,1 th testonly.lua -task vi_vggmpt_d_slcls $ITERATION_VGGM -stride 8 -diffLevel 1 -numGpu 2 -startFrom $DATA_OUT_DIR/vi_vggmpt_d_slcls,batchSize=384,epochSize=199,stride=8/model_026,batchSize=384,epochSize=199,learnRate=1e-4,1e-4,stride=8/model_019.t7 # 76.50
# Momentum network.
CUDA_VISIBLE_DEVICES=4,5,6,7 th main.lua -task vi_vggmpt_sdsel_slcls $ITERATION_VGGM -stride 1 -branchAfter 1 -mergeAfter 5 -diffChance 0.9 #(E22) 66.29
CUDA_VISIBLE_DEVICES=4,5,6,7 th main.lua -task vi_vggmpt_sdsel_slcls $ITERATION_VGGM -stride 1 -branchAfter 1 -mergeAfter 5 -diffChance 0.9 -learnRate 1e-4,1e-4 -startFrom $DATA_OUT_DIR/vi_vggmpt_sdsel_slcls,batchSize=384,diffChance=0.9,epochSize=199,mergeAfter=5,stride=1/model_022.t7 #(E22) 67.30
CUDA_VISIBLE_DEVICES=4,5 th testonly.lua -task vi_vggmpt_sdsel_slcls $ITERATION_VGGM -stride 1 -branchAfter 1 -mergeAfter 5 -diffChance 0.9 -numGpu 2 -startFrom $DATA_OUT_DIR/vi_vggmpt_sdsel_slcls,batchSize=384,diffChance=0.9,epochSize=199,mergeAfter=5,stride=1/model_022,batchSize=384,diffChance=0.9,epochSize=199,learnRate=1e-4,1e-4,mergeAfter=5,stride=1/model_022.t7 # 76.83
CUDA_VISIBLE_DEVICES=4,5,6,7 th main.lua -task vi_vggmpt_sdsel_slcls $ITERATION_VGGM -stride 2 -branchAfter 1 -mergeAfter 5 -diffChance 0.9 #(E25) 70.21
CUDA_VISIBLE_DEVICES=4,5,6,7 th main.lua -task vi_vggmpt_sdsel_slcls $ITERATION_VGGM -stride 2 -branchAfter 1 -mergeAfter 5 -diffChance 0.9 -learnRate 1e-4,1e-4 -startFrom $DATA_OUT_DIR/vi_vggmpt_sdsel_slcls,batchSize=384,diffChance=0.9,epochSize=199,mergeAfter=5/model_025.t7 #(E16) 71.32
CUDA_VISIBLE_DEVICES=4,5 th testonly.lua -task vi_vggmpt_sdsel_slcls $ITERATION_VGGM -stride 2 -branchAfter 1 -mergeAfter 5 -diffChance 0.9 -numGpu 2 -startFrom $DATA_OUT_DIR/vi_vggmpt_sdsel_slcls,batchSize=384,diffChance=0.9,epochSize=199,mergeAfter=5/model_025,batchSize=384,diffChance=0.9,epochSize=199,learnRate=1e-4,1e-4,mergeAfter=5/model_016.t7 # 78.88
CUDA_VISIBLE_DEVICES=4,5,6,7 th main.lua -task vi_vggmpt_sdsel_slcls $ITERATION_VGGM -stride 8 -branchAfter 1 -mergeAfter 5 -diffChance 0.9 #(E26) 72.06
CUDA_VISIBLE_DEVICES=4,5,6,7 th main.lua -task vi_vggmpt_sdsel_slcls $ITERATION_VGGM -stride 8 -branchAfter 1 -mergeAfter 5 -diffChance 0.9 -learnRate 1e-4,1e-4 -startFrom $DATA_OUT_DIR/vi_vggmpt_sdsel_slcls,batchSize=384,diffChance=0.9,epochSize=199,mergeAfter=5,stride=8/model_026.t7 #(E05) 72.53
CUDA_VISIBLE_DEVICES=4,5 th testonly.lua -task vi_vggmpt_sdsel_slcls $ITERATION_VGGM -stride 8 -branchAfter 1 -mergeAfter 5 -diffChance 0.9 -numGpu 2 -startFrom $DATA_OUT_DIR/vi_vggmpt_sdsel_slcls,batchSize=384,diffChance=0.9,epochSize=199,mergeAfter=5,stride=8/model_026,batchSize=384,diffChance=0.9,epochSize=199,learnRate=1e-4,1e-4,mergeAfter=5,stride=8/model_005.t7 # 78.73

#############################################################
### Per-frame prediction for video demonstration. ###########
### These results are also used to report test speeds. ######
# Spatial network.
CUDA_VISIBLE_DEVICES=0 th testonly-perframe.lua -task vi_vggmpt_s_slcls_pftest $ITERATION_VGGM -batchSize 256 -numGpu 1 -startFrom $DATA_OUT_DIR/vi_vggmpt_s_slcls,batchSize=384,epochSize=100/model_027,batchSize=384,epochSize=100,learnRate=1e-4,1e-4/model_030.t7
CUDA_VISIBLE_DEVICES=0 th testonly-perframe.lua -task vi_vgg16pt_s_slcls_pftest $ITERATION_VGG16 -batchSize 128 -numGpu 1 -startFrom $DATA_OUT_DIR/vi_vgg16pt_s_slcls,batchSize=256,epochSize=150/model_011,batchSize=256,epochSize=150,learnRate=1e-4,1e-4/model_018.t7
# Temporal network.
CUDA_VISIBLE_DEVICES=1 th testonly-perframe.lua -task vi_vggmpt_d_slcls_pftest $ITERATION_VGGM -batchSize 256 -stride 4 -diffLevel 1 -numGpu 1 -startFrom $DATA_OUT_DIR/vi_vggmpt_d_slcls,batchSize=384,epochSize=199,stride=4/model_024,batchSize=384,epochSize=199,learnRate=1e-4,1e-4,stride=4/model_026.t7
CUDA_VISIBLE_DEVICES=1 th testonly-perframe.lua -task vi_vgg16pt_d_slcls_pftest $ITERATION_VGG16 -batchSize 128 -stride 4 -diffLevel 4 -numGpu 1 -startFrom $DATA_OUT_DIR/vi_vgg16pt_d_slcls,batchSize=256,epochSize=299,stride=4/model_017,batchSize=256,epochSize=299,learnRate=1e-4,1e-4,stride=4/model_006.t7
# Momentum network.
CUDA_VISIBLE_DEVICES=2 th testonly-perframe.lua -task vi_vggmpt_sdsel_slcls_pftest $ITERATION_VGGM -batchSize 256 -stride 4 -branchAfter 1 -mergeAfter 5 -diffChance 0.9 -numGpu 1 -startFrom $DATA_OUT_DIR/vi_vggmpt_sdsel_slcls,batchSize=384,diffChance=0.9,epochSize=199,mergeAfter=5,stride=4/model_025,batchSize=384,diffChance=0.9,epochSize=199,learnRate=1e-4,1e-4,mergeAfter=5,stride=4/model_005.t7
CUDA_VISIBLE_DEVICES=2 th testonly-perframe.lua -task vi_vgg16pt_sdsel_slcls_pftest $ITERATION_VGG16 -batchSize 128 -stride 4 -branchAfter 4 -mergeAfter 13 -diffChance 0.9 -numGpu 1 -startFrom $DATA_OUT_DIR/vi_vgg16pt_sdsel_slcls,batchSize=192,epochSize=398,stride=4/model_023,batchSize=192,epochSize=398,learnRate=1e-4,1e-4,stride=4/model_030.t7

#############################################################
### Input-level gradient visualization. #####################
### (Suspended due to unsatisfactory results) ###############
# Spatial network.
CUDA_VISIBLE_DEVICES=0,1,2,3 th valgradonly.lua -task vi_vggmpt_s_slcls $ITERATION_VGGM -numDonkey 0 -startFrom $DATA_OUT_DIR/vi_vggmpt_s_slcls,batchSize=384,epochSize=100/model_027,batchSize=384,epochSize=100,learnRate=1e-4,1e-4/model_030.t7
CUDA_VISIBLE_DEVICES=0,1,2,3 th valgradonly.lua -task vi_vgg16pt_s_slcls $ITERATION_VGG16D -numDonkey 0 -startFrom $DATA_OUT_DIR/vi_vgg16pt_s_slcls,batchSize=256,epochSize=150/model_011,batchSize=256,epochSize=150,learnRate=1e-4,1e-4/model_018.t7
# Temporal network.
CUDA_VISIBLE_DEVICES=0,1,2,3 th valgradonly.lua -task vi_vggmpt_d_slcls $ITERATION_VGGM -stride 4 -numDonkey 0 -startFrom $DATA_OUT_DIR/vi_vggmpt_d_slcls,batchSize=384,epochSize=199,stride=4/model_024,batchSize=384,epochSize=199,learnRate=1e-4,1e-4,stride=4/model_026.t7
CUDA_VISIBLE_DEVICES=0,1,2,3 th valgradonly.lua -task vi_vgg16pt_d_slcls $ITERATION_VGG16D -stride 4 -numDonkey 0 -startFrom $DATA_OUT_DIR/vi_vgg16pt_d_slcls,batchSize=256,epochSize=299,stride=4/model_017,batchSize=256,epochSize=299,learnRate=1e-4,1e-4,stride=4/model_006.t7
