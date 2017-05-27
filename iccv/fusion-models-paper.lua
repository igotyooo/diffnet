require 'nn'
--local m = require 'matio'

local ucf1_vgg16_temat = '/home/dgyoo/workspace/dataout/diffnet/UCF101_MATLAB/ucf101-vgg16multiScaleRegular-bs=128-cheapRsz=0-split1-dr0.8/predictions.mat'
local ucf2_vgg16_temat = '/home/dgyoo/workspace/dataout/diffnet/UCF101_MATLAB/ucf101-vgg16multiScaleRegular-bs=128-cheapRsz=0-split2-dr0.8/predictions.mat'
local ucf3_vgg16_temat = '/home/dgyoo/workspace/dataout/diffnet/UCF101_MATLAB/ucf101-vgg16multiScaleRegular-bs=128-cheapRsz=0-split3-dr0.8/predictions.mat'
local ucf1_vgg16_spmat = ''
local ucf2_vgg16_spmat = ''
local ucf3_vgg16_spmat = ''
local ucf1_vgg16_sp = '/home/dgyoo/workspace/dataout/diffnet/UCF101_RGB_S1/vi_vgg16pt_s_slcls,batchSize=256,epochSize=150/model_011,batchSize=256,epochSize=150,learnRate=1e-4,1e-4/model_018,batchSize=192/test_nc25_1.t7'
local ucf2_vgg16_sp = '/home/dgyoo/workspace/dataout/diffnet_jolee/UCF101_RGB_S2/vi_vgg16pt_s_slcls,batchSize=256,epochSize=150/model_000,batchSize=256,epochSize=150,learnRate=1e-4,1e-4/model_000,batchSize=192/test_nc25_1.t7'
local ucf3_vgg16_sp = '/home/dgyoo/workspace/dataout/diffnet_jolee/UCF101_RGB_S3/vi_vgg16pt_s_slcls,batchSize=256,epochSize=150/model_000,batchSize=256,epochSize=150,learnRate=1e-4,1e-4/model_000,batchSize=192/test_nc25_1.t7'
local ucf1_vgg16_te = '/home/dgyoo/workspace/dataout/diffnet/UCF101_RGB_S1/vi_vgg16pt_d_slcls,batchSize=256,epochSize=299,stride=4/model_017,batchSize=256,epochSize=299,learnRate=1e-4,1e-4,stride=4/model_006,batchSize=192,stride=4/test_nc25_1.t7'
local ucf2_vgg16_te = '/home/dgyoo/workspace/dataout/diffnet_jolee/UCF101_RGB_S2/vi_vgg16pt_d_slcls,batchSize=256,epochSize=299,stride=4/model_000,batchSize=256,epochSize=299,learnRate=1e-4,1e-4,stride=4/model_000,batchSize=192/test_nc25_1.t7'
local ucf3_vgg16_te = '/home/dgyoo/workspace/dataout/diffnet_jolee/UCF101_RGB_S3/vi_vgg16pt_d_slcls,batchSize=256,epochSize=299,stride=4/model_000,batchSize=256,epochSize=299,learnRate=1e-4,1e-4,stride=4/model_000,batchSize=192/test_nc25_1.t7'
local ucf1_vgg16_mo = '/home/dgyoo/workspace/dataout/diffnet_jolee/UCF101_RGB_S1/vi_vgg16pt_sdsel_slcls,batchSize=192,diffChance=0.9,epochSize=398,stride=4/model_000,batchSize=192,diffChance=0.9,epochSize=398,learnRate=1e-4,1e-4,stride=4/model_000,batchSize=192,diffChance=0.9/test_nc25_1.t7'
local ucf2_vgg16_mo = '/home/dgyoo/workspace/dataout/diffnet_jolee/UCF101_RGB_S2/vi_vgg16pt_sdsel_slcls,batchSize=192,diffChance=0.9,epochSize=398,stride=4/model_000,batchSize=192,diffChance=0.9,epochSize=398,learnRate=1e-4,1e-4,stride=4/model_000,batchSize=192,diffChance=0.9/test_nc25_1.t7'
local ucf3_vgg16_mo = '/home/dgyoo/workspace/dataout/diffnet_jolee/UCF101_RGB_S3/vi_vgg16pt_sdsel_slcls,batchSize=192,diffChance=0.9,epochSize=398,stride=4/model_000,batchSize=192,diffChance=0.9,epochSize=398,learnRate=1e-4,1e-4,stride=4/model_000,batchSize=192,diffChance=0.9/test_nc25_1.t7'
local ucf1_vggm_sp = '/home/dgyoo/workspace/dataout/diffnet/UCF101_RGB_S1/vi_vggmpt_s_slcls,batchSize=384,epochSize=100/model_027,batchSize=384,epochSize=100,learnRate=1e-4,1e-4/model_030,batchSize=384/test_nc25_1.t7'
local ucf2_vggm_sp = '/home/dgyoo/workspace/dataout/diffnet_jolee/UCF101_RGB_S2/vi_vggmpt_s_slcls,batchSize=384,epochSize=100/model_000,batchSize=384,epochSize=100,learnRate=1e-4,1e-4/model_000,batchSize=192/test_nc25_1.t7'
local ucf3_vggm_sp = '/home/dgyoo/workspace/dataout/diffnet_jolee/UCF101_RGB_S3/vi_vggmpt_s_slcls,batchSize=384,epochSize=100/model_000,batchSize=384,epochSize=100,learnRate=1e-4,1e-4/model_000,batchSize=192/test_nc25_1.t7'
local ucf1_vggm_te = '/home/dgyoo/workspace/dataout/diffnet/UCF101_RGB_S1/vi_vggmpt_d_slcls,batchSize=384,epochSize=199,stride=4/model_024,batchSize=384,epochSize=199,learnRate=1e-4,1e-4,stride=4/model_026,batchSize=384,stride=4/test_nc25_1.t7'
local ucf2_vggm_te = '/home/dgyoo/workspace/dataout/diffnet_jolee/UCF101_RGB_S2/vi_vggmpt_d_slcls,batchSize=384,epochSize=199,stride=4/model_000,batchSize=384,epochSize=199,learnRate=1e-4,1e-4,stride=4/model_000,batchSize=192/test_nc25_1.t7'
local ucf3_vggm_te = '/home/dgyoo/workspace/dataout/diffnet_jolee/UCF101_RGB_S3/vi_vggmpt_d_slcls,batchSize=384,epochSize=199,stride=4/model_000,batchSize=384,epochSize=199,learnRate=1e-4,1e-4,stride=4/model_000,batchSize=192/test_nc25_1.t7'
local ucf1_vggm_mo = '/home/dgyoo/workspace/dataout/diffnet/UCF101_RGB_S1/vi_vggmpt_sdsel_slcls,batchSize=384,diffChance=0.9,epochSize=199,mergeAfter=5,stride=4/model_025,batchSize=384,diffChance=0.9,epochSize=199,learnRate=1e-4,1e-4,mergeAfter=5,stride=4/model_005,batchSize=384,diffChance=0.9,mergeAfter=5,stride=4/test_nc25_1.t7'
local ucf2_vggm_mo = '/home/dgyoo/workspace/dataout/diffnet_jolee/UCF101_RGB_S2/vi_vggmpt_sdsel_slcls,batchSize=384,diffChance=0.9,epochSize=199,mergeAfter=5,stride=4/model_000,batchSize=384,diffChance=0.9,epochSize=199,learnRate=1e-4,1e-4,mergeAfter=5,stride=4/model_000,batchSize=192,diffChance=0.9/test_nc25_1.t7'
local ucf3_vggm_mo = '/home/dgyoo/workspace/dataout/diffnet_jolee/UCF101_RGB_S3/vi_vggmpt_sdsel_slcls,batchSize=384,diffChance=0.9,epochSize=199,mergeAfter=5,stride=4/model_000,batchSize=384,diffChance=0.9,epochSize=199,learnRate=1e-4,1e-4,mergeAfter=5,stride=4/model_000,batchSize=192,diffChance=0.9/test_nc25_1.t7'
local ucf1_label = '/home/dgyoo/workspace/dataout/diffnet/UCF101_RGB_S1/dbVal.t7'
local ucf2_label = '/home/dgyoo/workspace/dataout/diffnet/UCF101_RGB_S2/dbVal.t7'
local ucf3_label = '/home/dgyoo/workspace/dataout/diffnet/UCF101_RGB_S3/dbVal.t7'

local hmdb1_vgg16_sp = '/home/dgyoo/workspace/dataout/diffnet_jolee/HMDB51_RGB_S1/vi_vgg16pt_s_slcls,batchSize=256,epochSize=150/model_000,batchSize=256,epochSize=150,learnRate=1e-4,1e-4/model_000,batchSize=192/test_nc25_1.t7'
local hmdb2_vgg16_sp = '/home/dgyoo/workspace/dataout/diffnet_jolee/HMDB51_RGB_S2/vi_vgg16pt_s_slcls,batchSize=256,epochSize=150/model_000,batchSize=256,epochSize=150,learnRate=1e-4,1e-4/model_000,batchSize=192/test_nc25_1.t7'
local hmdb3_vgg16_sp = '/home/dgyoo/workspace/dataout/diffnet_jolee/HMDB51_RGB_S3/vi_vgg16pt_s_slcls,batchSize=256,epochSize=150/model_000,batchSize=256,epochSize=150,learnRate=1e-4,1e-4/model_000,batchSize=192/test_nc25_1.t7'
local hmdb1_vgg16_te = '/home/dgyoo/workspace/dataout/diffnet_jolee/HMDB51_RGB_S1/vi_vgg16pt_d_slcls,batchSize=256,epochSize=299,stride=4/model_000,batchSize=256,epochSize=299,learnRate=1e-4,1e-4,stride=4/model_000,batchSize=192/test_nc25_1.t7'
local hmdb2_vgg16_te = '/home/dgyoo/workspace/dataout/diffnet_jolee/HMDB51_RGB_S2/vi_vgg16pt_d_slcls,batchSize=256,epochSize=299,stride=4/model_000,batchSize=256,epochSize=299,learnRate=1e-4,1e-4,stride=4/model_000,batchSize=192/test_nc25_1.t7'
local hmdb3_vgg16_te = '/home/dgyoo/workspace/dataout/diffnet_jolee/HMDB51_RGB_S3/vi_vgg16pt_d_slcls,batchSize=256,epochSize=299,stride=4/model_000,batchSize=256,epochSize=299,learnRate=1e-4,1e-4,stride=4/model_000,batchSize=192/test_nc25_1.t7'
local hmdb1_vgg16_mo = '/home/dgyoo/workspace/dataout/diffnet_jolee/HMDB51_RGB_S1/vi_vgg16pt_d_slcls,batchSize=256,epochSize=299,stride=4/model_000,batchSize=256,epochSize=299,learnRate=1e-4,1e-4,stride=4/model_000,batchSize=192/test_nc25_1.t7'
local hmdb2_vgg16_mo = '/home/dgyoo/workspace/dataout/diffnet_jolee/HMDB51_RGB_S2/vi_vgg16pt_d_slcls,batchSize=256,epochSize=299,stride=4/model_000,batchSize=256,epochSize=299,learnRate=1e-4,1e-4,stride=4/model_000,batchSize=192/test_nc25_1.t7'
local hmdb3_vgg16_mo = '/home/dgyoo/workspace/dataout/diffnet_jolee/HMDB51_RGB_S3/vi_vgg16pt_d_slcls,batchSize=256,epochSize=299,stride=4/model_000,batchSize=256,epochSize=299,learnRate=1e-4,1e-4,stride=4/model_000,batchSize=192/test_nc25_1.t7'
local hmdb1_vggm_sp =  '/home/dgyoo/workspace/dataout/diffnet_jolee/HMDB51_RGB_S1/vi_vggmpt_s_slcls,epochSize=75/model_000,epochSize=75,learnRate=1e-4,1e-4/model_000,batchSize=192/test_nc25_1.t7'
local hmdb2_vggm_sp =  '/home/dgyoo/workspace/dataout/diffnet_jolee/HMDB51_RGB_S2/vi_vggmpt_s_slcls,epochSize=75/model_000,epochSize=75,learnRate=1e-4,1e-4/model_000,batchSize=192/test_nc25_1.t7'
local hmdb3_vggm_sp =  '/home/dgyoo/workspace/dataout/diffnet_jolee/HMDB51_RGB_S3/vi_vggmpt_s_slcls,epochSize=75/model_000,epochSize=75,learnRate=1e-4,1e-4/model_000,batchSize=192/test_nc25_1.t7'
local hmdb1_vggm_te =  '/home/dgyoo/workspace/dataout/diffnet_jolee/HMDB51_RGB_S1/vi_vggmpt_d_slcls,epochSize=150,stride=4/model_000,epochSize=150,learnRate=1e-4,1e-4,stride=4/model_000,batchSize=192/test_nc25_1.t7'
local hmdb2_vggm_te =  '/home/dgyoo/workspace/dataout/diffnet_jolee/HMDB51_RGB_S2/vi_vggmpt_d_slcls,epochSize=150,stride=4/model_000,epochSize=150,learnRate=1e-4,1e-4,stride=4/model_000,batchSize=192/test_nc25_1.t7'
local hmdb3_vggm_te =  '/home/dgyoo/workspace/dataout/diffnet_jolee/HMDB51_RGB_S3/vi_vggmpt_d_slcls,epochSize=150,stride=4/model_000,epochSize=150,learnRate=1e-4,1e-4,stride=4/model_000,batchSize=192/test_nc25_1.t7'
local hmdb1_vggm_mo =  '/home/dgyoo/workspace/dataout/diffnet_jolee/HMDB51_RGB_S1/vi_vggmpt_sdsel_slcls,diffChance=0.9,epochSize=150,mergeAfter=5,stride=4/model_000,diffChance=0.9,epochSize=150,learnRate=1e-4,1e-4,mergeAfter=5,stride=4/model_000,batchSize=192,diffChance=0.9/test_nc25_1.t7'
local hmdb2_vggm_mo =  '/home/dgyoo/workspace/dataout/diffnet_jolee/HMDB51_RGB_S2/vi_vggmpt_sdsel_slcls,diffChance=0.9,epochSize=150,mergeAfter=5,stride=4/model_000,diffChance=0.9,epochSize=150,learnRate=1e-4,1e-4,mergeAfter=5,stride=4/model_000,batchSize=192,diffChance=0.9/test_nc25_1.t7'
local hmdb3_vggm_mo =  '/home/dgyoo/workspace/dataout/diffnet_jolee/HMDB51_RGB_S3/vi_vggmpt_sdsel_slcls,diffChance=0.9,epochSize=150,mergeAfter=5,stride=4/model_000,diffChance=0.9,epochSize=150,learnRate=1e-4,1e-4,mergeAfter=5,stride=4/model_000,batchSize=192,diffChance=0.9/test_nc25_1.t7'
local hmdb1_label =    '/home/dgyoo/workspace/dataout/diffnet/HMDB51_RGB_S1/dbVal.t7'
local hmdb2_label =    '/home/dgyoo/workspace/dataout/diffnet/HMDB51_RGB_S2/dbVal.t7'
local hmdb3_label =    '/home/dgyoo/workspace/dataout/diffnet/HMDB51_RGB_S3/dbVal.t7'

local srcDbPath = ucf1_label
local sid2path = { ucf1_vgg16_sp, ucf1_vgg16_te, ucf1_vgg16_mo }
local resolution = 20

local function loadTensor( path )
    if path:match( '.+%.mat' ) then
        local t = m.load( path, 'output' ):transpose( 1, 2 )
        t = nn.LogSoftMax(  ):forward( t:double(  ) )
        return t:float(  )
    else
        return torch.load( path )
    end
end
local function evaluate( vid2pred, db )
    local numQuery = vid2pred:size( 1 )
	local numClass = db.cid2name:size( 1 )
    local vid2top1 = torch.Tensor( numQuery ):fill( 0 )
    local cid2num = torch.Tensor( numClass ):fill( 0 )
    local cid2top1 = torch.Tensor( numClass ):fill( 0 )
    local _, vid2rank2cid = vid2pred:float(  ):sort( 2, true )
    for vid = 1, numQuery do
        local pcid = vid2rank2cid[ vid ][ 1 ]
        local cid = db.vid2cid[ vid ]
        local score = 0
        if pcid == cid then score = 1 end
        vid2top1[ vid ] = score
        cid2top1[ cid ] = cid2top1[ cid ] + score
        cid2num[ cid ] = cid2num[ cid ] + 1
    end
    cid2top1:cdiv( cid2num )
    return vid2top1:mean(  ) * 100, cid2top1:mean(  ) * 100
end

local db = torch.load( srcDbPath )
local sid2vid2pred = {  }
for k,v in pairs( sid2path ) do
    sid2vid2pred[ k ] = loadTensor( v )
end
local bestw, besttop1, bestacc = torch.Tensor( #sid2path - 1, 2 ), 0, 0
local predPrev = sid2vid2pred[ 1 ]:clone(  )
for c = 2, #sid2path do
    local predCurr = sid2vid2pred[ c ]
    local maxw, maxtop1, maxacc = 0, 0, 0
    for r = 0, resolution do
        local w = r / resolution
        local top1, acc = evaluate( ( 1 - w ) * predPrev + w * predCurr, db )
        if maxtop1 < top1 then 
            maxw = w
            maxtop1 = top1
        end
        if maxacc < acc then maxacc = acc end
        print( string.format( 'FUSION%d) [%.2f, %.2f] Top1 %.2f, Acc %.2f', c - 1, 1 - w, w, top1, acc ) )
    end
    print( string.format( 'FUSION%d) [%.2f, %.2f] Top1 %.2f, Acc %.2f -- BEST', c - 1, 1 - maxw, maxw, maxtop1, maxacc ) )
    predPrev = ( 1 - maxw ) * predPrev + maxw * predCurr
    if besttop1 < maxtop1 then 
        bestw[ c - 1 ][ 1 ] = 1 - maxw
        bestw[ c - 1 ][ 2 ] = maxw
        besttop1 = maxtop1
    end
    if bestacc < maxacc then bestacc = maxacc end
end
print( string.format( '--\nTop1 %.2f, Acc %.2f -- BEST OF THE BEST', besttop1, bestacc ) )
print( bestw )
