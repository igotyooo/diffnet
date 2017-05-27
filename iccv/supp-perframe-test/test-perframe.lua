local test = {  }
test.queryBatch = torch.CudaTensor(  )
test.netTimer = torch.Timer(  )
test.dataTimer = torch.Timer(  )
function test.setOption( opt, queries )
	assert( opt.batchSize > 0 )
	assert( opt.numOut > 0 )
    local tmp1, tmp2, i = {  }, {  }, 0
    for k,v in pairs( queries ) do tmp1[ v ] = v end
    for k,v in pairs( tmp1 ) do
        i = i + 1
        tmp2[ i ] = v
    end
    queries = tmp2
	test.batchSize = opt.batchSize
	test.queries = queries
    test.numQuery = #queries
end
function test.setModel( model )
	test.model = model.model
end
function test.setDonkey( donkeys )
	test.donkeys = donkeys
end
function test.setFunction( changeModel, getQuery, aggregateAnswers, evaluate )
	test.changeModel = changeModel
	test.getQuery = getQuery
	test.aggregateAnswers = aggregateAnswers
	test.evaluate = evaluate
end
function test.test(  )
	-- Initialization.
	local getQuery = test.getQuery
	local answerQuery = test.answerQuery
	test.queryNumber = 0
	-- Do the job.
	test.print( 'Test.' )
	cutorch.synchronize(  )
	test.changeModel( test.model )
	test.model:evaluate(  )
	for q, qid in pairs( test.queries ) do
		test.donkeys:addjob(
			function(  )
				return getQuery( qid ), qid
			end, -- Job callback.
			function( x, y )
				answerQuery( x, y )
			end -- End callback.
		)
	end
	test.donkeys:synchronize(  )
	cutorch.synchronize(  )
	collectgarbage(  )
end
function test.answerQuery( query, qid )
	local dataTime = test.dataTimer:time(  ).real
	local querySize = query:size( 1 )
	local batchSize = test.batchSize
	local numBatch = math.max( 1, math.ceil( querySize / batchSize ) )
	local answers
	test.netTimer:reset(  )
	for b = 1, numBatch do
		-- Make a query batch.
		local s = ( b - 1 ) * batchSize + 1
		local n = math.min( querySize - s + 1, batchSize )
		local queryBatch = query:narrow( 1, s, n )
		test.queryBatch:resize( queryBatch:size(  ) ):copy( queryBatch )
		-- Forward pass.
		cutorch.synchronize(  )
		local answerBatch = test.model:forward( test.queryBatch )
		cutorch.synchronize(  )
		-- Move back to cpu.
		answerBatchCpu = {  }
		if type( answerBatch ) ~= 'table' then answerBatch = { answerBatch } end
		for k, v in pairs( answerBatch ) do answerBatchCpu[ k ] = v:float(  ) end
		-- Concatenate answers.
		answers = test.concatenate( answers, answerBatchCpu )
		collectgarbage(  )
	end
	-- Aggregate answers.
	local answer = test.aggregateAnswers( answers, qid )
	local netTime = test.netTimer:time(  ).real
	local totalTime = dataTime + netTime
	local speed = querySize / totalTime
	test.queryNumber = test.queryNumber + 1
	-- Print information.
	test.print( string.format( '%d/%d, qid %d, %dim/s (data %.2fs, net %.2fs)', 
		test.queryNumber, test.numQuery, qid, speed, dataTime, netTime ) )
	test.dataTimer:reset(  )
	collectgarbage(  )
end
function test.concatenate( table1, table2 )
	if table1 then
		for k, v in pairs( table2 ) do
			table1[ k ] = table1[ k ]:cat( v, 1 )
		end
	else
		table1 = table2
	end
	return table1
end
function test.print( str )
	print( 'TEST) ' .. str )
end
return test
