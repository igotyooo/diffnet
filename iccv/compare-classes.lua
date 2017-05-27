require 'torch'
require 'nn'
local ffi = require 'ffi'
local ucf1_vgg16_sp = '/home/dgyoo/workspace/dataout/diffnet/UCF101_RGB_S1/vi_vgg16pt_s_slcls,batchSize=256,epochSize=150/model_011,batchSize=256,epochSize=150,learnRate=1e-4,1e-4/model_018,batchSize=192/test_nc25_1.t7'
local ucf1_vgg16_te = '/home/dgyoo/workspace/dataout/diffnet/UCF101_RGB_S1/vi_vgg16pt_d_slcls,batchSize=256,epochSize=299,stride=4/model_017,batchSize=256,epochSize=299,learnRate=1e-4,1e-4,stride=4/model_006,batchSize=192,stride=4/test_nc25_1.t7'
--local ucf1_vgg16_mo = '/home/dgyoo/workspace/dataout/diffnet_dataout/UCF101_RGB_S1/vi_vgg16pt_sdsel_slcls,batchSize=192,diffChance=0.9,epochSize=398,stride=4/model_000,batchSize=192,diffChance=0.9,epochSize=398,learnRate=1e-4,1e-4,stride=4/model_000,batchSize=192,diffChance=0.9/test_nc25_1.t7'
local ucf1_vgg16_mo = '/home/dgyoo/workspace/dataout/diffnet/UCF101_RGB_S1/vi_vgg16pt_sdsel_slcls,batchSize=192,epochSize=398,stride=4/model_023,batchSize=192,epochSize=398,learnRate=1e-4,1e-4,stride=4/model_030,batchSize=192,diffChance=0.9,stride=4/test_nc25_1.t7'
local ucf1_vggm_sp = '/home/dgyoo/workspace/dataout/diffnet/UCF101_RGB_S1/vi_vggmpt_s_slcls,batchSize=384,epochSize=100/model_027,batchSize=384,epochSize=100,learnRate=1e-4,1e-4/model_030,batchSize=384/test_nc25_1.t7'
local ucf1_vggm_te = '/home/dgyoo/workspace/dataout/diffnet/UCF101_RGB_S1/vi_vggmpt_d_slcls,batchSize=384,epochSize=199,stride=4/model_024,batchSize=384,epochSize=199,learnRate=1e-4,1e-4,stride=4/model_026,batchSize=384,stride=4/test_nc25_1.t7'
local ucf1_vggm_mo = '/home/dgyoo/workspace/dataout/diffnet/UCF101_RGB_S1/vi_vggmpt_sdsel_slcls,batchSize=384,diffChance=0.9,epochSize=199,mergeAfter=5,stride=4/model_025,batchSize=384,diffChance=0.9,epochSize=199,learnRate=1e-4,1e-4,mergeAfter=5,stride=4/model_005,batchSize=384,diffChance=0.9,mergeAfter=5,stride=4/test_nc25_1.t7'
local ucf1_label = '/home/dgyoo/workspace/dataout/diffnet/UCF101_RGB_S1/dbVal.t7'

local pathTarget = ucf1_vgg16_mo
local pathRefer = ucf1_vgg16_sp
local srcDbPath = ucf1_label
local numTopClass = 10

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
    local confmat = torch.Tensor( numClass, numClass ):fill( 0 )
    local _, vid2rank2cid = vid2pred:float(  ):sort( 2, true )
    for vid = 1, numQuery do
        local pcid = vid2rank2cid[ vid ][ 1 ]
        local cid = db.vid2cid[ vid ]
        local score = 0
        if pcid == cid then score = 1 end
        confmat[ cid ][ pcid ] = confmat[ cid ][ pcid ] + 1
        vid2top1[ vid ] = score
        cid2top1[ cid ] = cid2top1[ cid ] + score
        cid2num[ cid ] = cid2num[ cid ] + 1
    end
    cid2top1:cdiv( cid2num )
    for cid = 1, numClass do
        confmat[ cid ]:mul( 1 / cid2num[ cid ] )
    end
    return vid2top1, cid2top1, confmat
end
local function printConfmat( confmat )
    local numCls = confmat:size( 1 )
    for i = 1, numCls do
        local str = ''
        for j = 1, numCls do
            if j < numCls then
                str = str .. string.format( '%.0f ', confmat[ i ][ j ] * 100 )
            else
                str = str .. string.format( '%.0f;', confmat[ i ][ j ] * 100 )
            end
        end
        print( str )
    end
end

local db = torch.load( srcDbPath )
local numVid = db.vid2cid:numel(  )
local numCls = db.cid2name:size( 1 )
local vid2trueTar, cid2top1Tar, confmatTar = evaluate( loadTensor( pathTarget ), db )
local vid2trueRef, cid2top1Ref, confmatRef = evaluate( loadTensor( pathRefer ), db )
local vid2win = vid2trueTar - vid2trueRef
local cid2win = cid2top1Tar - cid2top1Ref
local rank2gap, rank2cid = cid2win:sort( 1, true )
local cid2name = {  }
for c = 1, numCls do
    cid2name[ c ] = ffi.string( torch.data( db.cid2name[ c ] ) )
end
local vidToShow = {  }
for r = 1, numTopClass do
    local cid = rank2cid[ r ]
    local gap = rank2gap[ r ]
    local cname = cid2name[ cid ]
    vidToShow[ r ] = torch.linspace( 1, numVid, numVid )[ torch.cmul( db.vid2cid:eq( cid ), vid2win:gt( 0 ) ) ]
    print( string.format( '%02d %.1f %.1f %s', r, cid2top1Tar[ cid ] * 100, cid2top1Ref[ cid ] * 100, cname ) )
end
print( '' )
for r = 1, numTopClass do
    local cid = rank2cid[ -r ]
    local gap = rank2gap[ -r ]
    local cname = cid2name[ cid ]
    print( string.format( '%02d %.1f %.1f %s', 102 - r, cid2top1Tar[ cid ] * 100, cid2top1Ref[ cid ] * 100, cname ) )
end
print( '' )
local str = ''
for k,v in pairs( vidToShow ) do
    for i = 1, v:numel(  ) do
        str = str .. v[ i ] .. ','
    end
end
print( 'IDs to show.' )
print( str )
print( '' )
print( 'Confmat of target' )
printConfmat( confmatTar )
print( '' )
print( 'Confmat of Reference' )
printConfmat( confmatRef )
print( '' )
print( 'Classes.' )
for c = 1, numCls do
    print( string.format( '\'%s\';', ffi.string( torch.data( db.cid2name[ c ] ) ) ) )
end
