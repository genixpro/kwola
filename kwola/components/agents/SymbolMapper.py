import copy
import io
import numpy
import os.path
import pickle
import pprint
import torch
import scipy.sparse
from ...config.logger import getLogger
import traceback
import matplotlib.pyplot as plt

class LineOfCodeSymbolMapping:
    def __init__(self, branchTrace, recentSymbolIndex, coverageSymbolIndex):
        self.branchTrace = {
            fileName: SymbolMapper.createSparseTraceArray(trace)
            for fileName, trace in branchTrace.items()
        }

        for sparseArray in self.branchTrace.values():
            sparseArray.eliminate_zeros()

        self.recentSymbolIndex = recentSymbolIndex
        self.coverageSymbolIndex = coverageSymbolIndex
        self.totalTracesWithSymbol = 0
        self.tracesSinceLastSeen = 0

    def linesOfCodeMatched(self):
        total = 0
        for fileName in self.branchTrace.keys():
            total += numpy.count_nonzero(self.branchTrace[fileName])
        return total


    def __repr__(self):
        return pprint.pformat({fileName: list(numpy.nonzero(trace)[0]) for fileName, trace in self.branchTrace.items()})

class SymbolMapper:
    def __init__(self, config):
        self.knownFiles = set()

        self.nextSymbolIndex = 1

        self.symbolMap = {

        }

        self.allSymbols = []

        self.modelFileName = "deep_learning_model"
        self.symbolMapFileName = "symbol_mapper"

        self.config = config

    @staticmethod
    def branchCoveredSymbol(fileName, branchIndex):
        return f'{branchIndex}-{fileName}-covered-branch'

    @staticmethod
    def branchRecentlyExecutedSymbol(fileName, branchIndex):
        return f'{branchIndex}-{fileName}-recently-executed-branch'

    def load(self):
        # We also need to load the symbol map - this is the mapping between symbol strings
        # and their index values within the embedding structure
        symbolMapData = self.config.loadKwolaFileData("models", self.symbolMapFileName, printErrorOnFailure=False)
        if symbolMapData is not None:
            (self.symbolMap, self.knownFiles, self.nextSymbolIndex, self.allSymbols) = pickle.loads(symbolMapData)

    def save(self):
        fileData = pickle.dumps((self.symbolMap, self.knownFiles, self.nextSymbolIndex, self.allSymbols), protocol=pickle.HIGHEST_PROTOCOL)
        self.config.saveKwolaFileData("models", self.symbolMapFileName, fileData)


    def findNextLOCSymbolMapping(self, fileName, branchTrace):
        branchIndex = None
        locSymbolMapping = None
        for value, branchIndex in zip(branchTrace.data, branchTrace.indices):
            if value >= 0:
                locSymbolMapping = self.symbolMap.get((fileName, branchIndex))
                if locSymbolMapping is not None:
                    break

        if locSymbolMapping is None:
            return None, None
        else:
            return locSymbolMapping, branchIndex

    def getAllLOCSymbolMappingsForBranchTrace(self, branchTrace):
        locSymbols = []
        locBranchValues = []
        locFileNames = []
        localBranchTraces = {}
        localBranchTraceValues = {}

        for fileName in branchTrace:
            local = SymbolMapper.createSparseTraceArray(branchTrace[fileName])
            localBranchTraceValues[fileName] = dict(zip(local.indices, local.data))
            local = local.minimum(1).maximum(0).astype(numpy.bool).astype(numpy.float64)
            localBranchTraces[fileName] = set(local.indices)

        for fileName in localBranchTraces.keys():
            while len(localBranchTraces[fileName]):
                branchIndex = None
                locSymbolMapping = None
                while len(localBranchTraces[fileName]):
                    branchIndex = localBranchTraces[fileName].pop()
                    locSymbolMapping = self.symbolMap.get((fileName, branchIndex))
                    if locSymbolMapping is not None:
                        break

                if locSymbolMapping is not None:
                    for locTraceFileName in locSymbolMapping.branchTrace:
                        if locTraceFileName in localBranchTraces:
                            localBranchTraces[locTraceFileName].difference_update(locSymbolMapping.branchTrace[locTraceFileName].indices)

                    locSymbols.append(locSymbolMapping)
                    locBranchValues.append(localBranchTraceValues[fileName][branchIndex])
                    locFileNames.append(fileName)
                else:
                    break

        return locSymbols, locBranchValues, locFileNames

    def computeCodePrevalenceScores(self, executionTraces):
        allTraceSymbolCounts = []

        for trace in executionTraces:
            locSymbols, locBranchValues, locFileNames = self.getAllLOCSymbolMappingsForBranchTrace(trace.branchTrace)

            symbolCounts = [locSymbolMapping.totalTracesWithSymbol for locSymbolMapping in locSymbols]

            if len(symbolCounts) == 0:
                allTraceSymbolCounts.append(-1)
            else:
                allTraceSymbolCounts.append(numpy.mean(symbolCounts))

        sortedSymbolCounts = sorted([count for count in allTraceSymbolCounts if count != -1])

        for trace, symbolCount in zip(executionTraces, allTraceSymbolCounts):
            if symbolCount == -1:
                trace.codePrevalenceScore = None
            else:
                trace.codePrevalenceScore = float(sortedSymbolCounts.index(symbolCount) / len(sortedSymbolCounts))

    def computeCoverageSymbolsList(self, executionTrace, beforeOrAfter):
        symbolIndexes = []
        weights = []

        if beforeOrAfter == "before":
            locSymbols, locBranchValues, locFileNames = self.getAllLOCSymbolMappingsForBranchTrace(executionTrace.cachedStartCumulativeBranchTrace)
        elif beforeOrAfter == "after":
            locSymbols, locBranchValues, locFileNames = self.getAllLOCSymbolMappingsForBranchTrace(executionTrace.cachedEndCumulativeBranchTrace)

        for locSymbolMapping in locSymbols:
            symbolIndexes.append(locSymbolMapping.coverageSymbolIndex)
            weights.append(1.0)

        return symbolIndexes, weights


    def computeDecayingBranchTraceSymbolsList(self, executionTrace, beforeOrAfter):
        symbolIndexes = []
        weights = []

        locSymbols, locBranchValues, locFileNames = None, None, None

        if beforeOrAfter == "before":
            locSymbols, locBranchValues, locFileNames = self.getAllLOCSymbolMappingsForBranchTrace(executionTrace.cachedStartDecayingBranchTrace)
        elif beforeOrAfter == "after":
            locSymbols, locBranchValues, locFileNames = self.getAllLOCSymbolMappingsForBranchTrace(executionTrace.cachedEndDecayingBranchTrace)

        for locSymbolMapping, branchValue, fileName in zip(locSymbols, locBranchValues, locFileNames):
            symbolIndexes.append(locSymbolMapping.recentSymbolIndex)
            weights.append(float(branchValue))

        return symbolIndexes, weights


    def computeDecayingFutureBranchTraceSymbolsList(self, executionTrace, beforeOrAfter):
        symbolIndexes = []
        weights = []

        locSymbols, locBranchValues, locFileNames = None, None, None

        if beforeOrAfter == "before":
            locSymbols, locBranchValues, locFileNames = self.getAllLOCSymbolMappingsForBranchTrace(executionTrace.cachedStartDecayingFutureBranchTrace)
        elif beforeOrAfter == "after":
            locSymbols, locBranchValues, locFileNames = self.getAllLOCSymbolMappingsForBranchTrace(executionTrace.cachedEndDecayingFutureBranchTrace)

        for locSymbolMapping, branchValue, fileName in zip(locSymbols, locBranchValues, locFileNames):
            symbolIndexes.append(locSymbolMapping.recentSymbolIndex)
            weights.append(float(branchValue))

        return symbolIndexes, weights


    def computeAllSymbolsForTrace(self, executionTrace, place):
        allSymbolList = []
        allWeightList = []

        symbols, weights = self.computeCoverageSymbolsList(executionTrace, place)
        allSymbolList.extend(symbols)
        allWeightList.extend(weights)

        symbols, weights = self.computeDecayingBranchTraceSymbolsList(executionTrace, place)
        allSymbolList.extend(symbols)
        allWeightList.extend(weights)

        return allSymbolList, allWeightList

    def assignNewSymbols(self, executionTraces):
        """
            This method will go through all of the execution traces provided, and for each one,
            it will check to see if there were any new symbols seen that need to be assigned.
            Symbols can be anything from a line of code being executed through to an interaction
            with a particular path or url or variable name. Basically they are a way of giving the
            model an indication of what state its in and what has happened recently, visa via
            these symbols which become neural network embeddings.

            :param executionTraces: A list or generator providing kwola.datamodels.ExecutionTraceModel objects

            :return: An integer providing the number of new symbols added


        """
        getLogger().info(f"Assigning symbols for {len(executionTraces)} traces")

        fileData = self.config.loadKwolaFileData("models", self.modelFileName, printErrorOnFailure=False)
        buffer = io.BytesIO(fileData)

        # Depending on whether GPU is turned on, we try load the state dict
        # directly into GPU / CUDA memory.
        stateDict = torch.load(buffer, map_location=torch.device('cpu'))
        symbolEmbeddingTensor = numpy.array(stateDict['symbolEmbedding.weight'].data)
        embeddingSize = symbolEmbeddingTensor.shape[1]

        newSymbolMaps = []
        newSymbolMapAssociatedOriginalSymbolMaps = []
        removedSymbolMaps = []

        splitSymbolsCount = 0
        netNewSymbolsCount = 0

        # Increment the value for tracesSinceLastSeen in all of the symbols.
        # If a given symbol is actually used in this set of traces, then the
        # value gets reset back to 0 below.
        for symbol in self.allSymbols:
            if not hasattr(symbol, 'tracesSinceLastSeen'):
                symbol.tracesSinceLastSeen = 0
            if not hasattr(symbol, 'totalTracesWithSymbol'):
                symbol.totalTracesWithSymbol = 0

            symbol.tracesSinceLastSeen += len(executionTraces)

        for trace in executionTraces:
            localBranchTraceIndexSets = {}

            for fileName in trace.branchTrace.keys():
                local = SymbolMapper.createSparseTraceArray(trace.branchTrace[fileName])
                local = local.minimum(1).maximum(0).astype(numpy.bool).astype(numpy.float64)
                localBranchTraceIndexSets[fileName] = set(local.indices)

            createNewLOCMap = False

            symbolMapFound = True
            while symbolMapFound:
                symbolMapFound = False
                for fileName in localBranchTraceIndexSets.keys():
                    if len(localBranchTraceIndexSets[fileName]) > 0:
                        branchIndex = None
                        locSymbolMapping = None
                        for branchIndex in localBranchTraceIndexSets[fileName]:
                            locSymbolMapping = self.symbolMap.get((fileName, branchIndex))
                            if locSymbolMapping is not None:
                                break

                        if locSymbolMapping is not None:
                            symbolMapFound = True

                            locSymbolMapping.totalTracesWithSymbol += 1
                            locSymbolMapping.tracesSinceLastSeen = 0

                            negativeBranchTraceIndexSets = {}
                            for locSymbolMapFileName in locSymbolMapping.branchTrace.keys():
                                if locSymbolMapFileName not in localBranchTraceIndexSets:
                                    localBranchTraceIndexSets[locSymbolMapFileName] = set()

                                negatives = set(locSymbolMapping.branchTrace[locSymbolMapFileName].indices)
                                negatives.difference_update(localBranchTraceIndexSets[locSymbolMapFileName])

                                localBranchTraceIndexSets[locSymbolMapFileName].difference_update(locSymbolMapping.branchTrace[locSymbolMapFileName].indices)

                                if len(negatives) > 0:
                                    negativeBranchTraceIndexSets[locSymbolMapFileName] = negatives

                            if len(negativeBranchTraceIndexSets):
                                splitSymbolsCount += 1

                                # We have to split the previous line of code symbol mapping into two separate mappings, because those lines of code have been observed
                                # to occur separately
                                firstNewSymbolMapDict = {}
                                for locSymbolMapFileName, branchTrace in locSymbolMapping.branchTrace.items():
                                    indices = list(set(branchTrace.indices).difference(negativeBranchTraceIndexSets.get(locSymbolMapFileName, set())))

                                    if len(indices) > 0:
                                        firstNewSymbolMapDict[locSymbolMapFileName] = scipy.sparse.csc_matrix(
                                            (numpy.ones(len(indices)),
                                             (indices, numpy.zeros(len(indices), dtype=numpy.int32))),
                                            shape=locSymbolMapping.branchTrace[locSymbolMapFileName].shape
                                        )

                                firstNewSymbolMap = LineOfCodeSymbolMapping(firstNewSymbolMapDict, None, None)

                                secondNewSymbolMap = LineOfCodeSymbolMapping({
                                    locSymbolMapFileName: scipy.sparse.csc_matrix(
                                        ( numpy.ones(len(branchTrace)), (list(branchTrace), numpy.zeros(len(branchTrace), dtype=numpy.int32)) ),
                                        shape=locSymbolMapping.branchTrace[locSymbolMapFileName].shape
                                    )
                                    for locSymbolMapFileName, branchTrace in negativeBranchTraceIndexSets.items()
                                }, None, None)

                                firstNewSymbolMap.totalTracesWithSymbol = locSymbolMapping.totalTracesWithSymbol
                                secondNewSymbolMap.totalTracesWithSymbol = locSymbolMapping.totalTracesWithSymbol

                                firstNewSymbolMap.tracesSinceLastSeen = 0
                                secondNewSymbolMap.tracesSinceLastSeen = 0

                                self.insertLOCSymbolMap(firstNewSymbolMap)
                                self.insertLOCSymbolMap(secondNewSymbolMap)

                                if locSymbolMapping.recentSymbolIndex is None:
                                    index = newSymbolMaps.index(locSymbolMapping)
                                    del newSymbolMaps[index]
                                    del newSymbolMapAssociatedOriginalSymbolMaps[index]
                                else:
                                    removedSymbolMaps.append(locSymbolMapping)

                                newSymbolMaps.append(firstNewSymbolMap)
                                newSymbolMaps.append(secondNewSymbolMap)

                                newSymbolMapAssociatedOriginalSymbolMaps.append(locSymbolMapping)
                                newSymbolMapAssociatedOriginalSymbolMaps.append(locSymbolMapping)
                        else:
                            createNewLOCMap = True

                    if symbolMapFound:
                        break

            if createNewLOCMap:
                netNewSymbolsCount += 1
                newSymbolBranchTrace = {}

                for fileName in localBranchTraceIndexSets.keys():
                    if len(localBranchTraceIndexSets[fileName]):
                        branchTrace = scipy.sparse.csc_matrix(
                            (numpy.ones(len(localBranchTraceIndexSets[fileName])), (list(localBranchTraceIndexSets[fileName]), numpy.zeros(len(localBranchTraceIndexSets[fileName]), dtype=numpy.int32)) ),
                            shape=trace.branchTrace[fileName].shape
                        )
                        newSymbolBranchTrace[fileName] = branchTrace

                newSymbolMap = LineOfCodeSymbolMapping(newSymbolBranchTrace, None, None)

                self.insertLOCSymbolMap(newSymbolMap)

                newSymbolMaps.append(newSymbolMap)
                newSymbolMapAssociatedOriginalSymbolMaps.append(None)


        for newLocSymbolMapping, oldLocSymbolMapping in zip(newSymbolMaps, newSymbolMapAssociatedOriginalSymbolMaps):
            nextSymbolIndex = self.nextSymbolIndex
            self.nextSymbolIndex += 1

            if self.nextSymbolIndex >= self.config['symbol_dictionary_size']:
                break

            if oldLocSymbolMapping is not None and oldLocSymbolMapping.recentSymbolIndex is not None:
                # When splitting a symbol mapping, we base the new tensor on the original tensor, but add in 20% random noise
                # so it can partially differentiate from the other child tensor. NOTE: We should actually test this at some
                # point and measure whether its better to reset the tensor completely randomly or derive it like this.
                origTensor = symbolEmbeddingTensor[oldLocSymbolMapping.recentSymbolIndex]
                symbolEmbeddingTensor[nextSymbolIndex] = origTensor + numpy.random.normal(0, numpy.std(origTensor) * 0.2, size=embeddingSize)
            else:
                # Create a blank random tensor for the new symbol
                symbolEmbeddingTensor[nextSymbolIndex] = numpy.random.normal(0, 1, size=embeddingSize)

            newLocSymbolMapping.recentSymbolIndex = nextSymbolIndex

            nextSymbolIndex = self.nextSymbolIndex
            self.nextSymbolIndex += 1

            if self.nextSymbolIndex >= self.config['symbol_dictionary_size']:
                break

            if oldLocSymbolMapping is not None and oldLocSymbolMapping.coverageSymbolIndex is not None:
                # When splitting a symbol mapping, we base the new tensor on the original tensor, but add in 20% random noise
                # so it can partially differentiate from the other child tensor. NOTE: We should actually test this at some
                # point and measure whether its better to reset the tensor completely randomly or derive it like this.
                origTensor = symbolEmbeddingTensor[oldLocSymbolMapping.coverageSymbolIndex]
                symbolEmbeddingTensor[nextSymbolIndex] = origTensor + numpy.random.normal(0, numpy.std(origTensor) * 0.2, size=embeddingSize)
            else:
                # Create a blank random tensor for the new symbol
                symbolEmbeddingTensor[nextSymbolIndex] = numpy.random.normal(0, 1, size=embeddingSize)

            newLocSymbolMapping.coverageSymbolIndex = nextSymbolIndex

            for fileName in newLocSymbolMapping.branchTrace.keys():
                self.knownFiles.add(fileName)

            self.allSymbols.append(newLocSymbolMapping)


        for newLocSymbolMapping in removedSymbolMaps:
            self.allSymbols.remove(newLocSymbolMapping)

        stateDict['symbolEmbedding.weight'] = torch.tensor(symbolEmbeddingTensor, dtype=stateDict['symbolEmbedding.weight'].dtype)

        buffer = io.BytesIO()
        torch.save(stateDict, buffer)
        self.config.saveKwolaFileData("models", self.modelFileName, buffer.getvalue())

        self.validateSymbolMaps()

        getLogger().info(f"New: {netNewSymbolsCount}, split: {splitSymbolsCount}")

        return netNewSymbolsCount, splitSymbolsCount


    def insertLOCSymbolMap(self, newSymbolMap):
        for locSymbolMapFileName in newSymbolMap.branchTrace.keys():
            nonZeroIndexes = numpy.nonzero(newSymbolMap.branchTrace[locSymbolMapFileName])[0]
            for branchIndex in nonZeroIndexes:
                self.symbolMap[(locSymbolMapFileName, branchIndex)] = newSymbolMap


    def validateSymbolMaps(self):
        all = set()
        for symbol in self.allSymbols:
            for fileName in symbol.branchTrace.keys():
                indexes = numpy.nonzero(symbol.branchTrace[fileName])[0]
                for index in indexes:
                    key = (fileName, index)
                    if key in all:
                        raise ValueError(f"The symbol map is invalid - there was an overlapping line of code mapping {key}")
                    else:
                        all.add(key)

    def computeCachedCumulativeBranchTraces(self, executionTraces):
        if len(executionTraces) == 0:
            return

        executionTraces[0].cachedStartCumulativeBranchTrace = {
            name: SymbolMapper.createSparseTraceArray(scipy.sparse.csc_matrix(value.shape, dtype=numpy.float64))
            for name, value in executionTraces[0].branchTrace.items()
        }

        executionTraces[0].cachedEndCumulativeBranchTrace = {
            name: SymbolMapper.createSparseTraceArray(value)
            for name, value in executionTraces[0].branchTrace.items()
        }

        lastTrace = executionTraces[0]

        for trace in executionTraces[1:]:
            if trace.cachedStartCumulativeBranchTrace is None:
                trace.cachedStartCumulativeBranchTrace = copy.deepcopy(lastTrace.cachedEndCumulativeBranchTrace)
                trace.cachedEndCumulativeBranchTrace = copy.deepcopy(lastTrace.cachedEndCumulativeBranchTrace)

                for fileName in trace.branchTrace.keys():
                    if fileName in trace.cachedEndCumulativeBranchTrace:
                        if trace.branchTrace[fileName].shape[0] == trace.cachedEndCumulativeBranchTrace[fileName].shape[0]:
                            trace.cachedEndCumulativeBranchTrace[fileName] += trace.branchTrace[fileName]
                    else:
                        trace.cachedEndCumulativeBranchTrace[fileName] = trace.branchTrace[fileName]
            lastTrace = trace


    def computeCachedDecayingBranchTrace(self, executionTraces):
        if len(executionTraces) == 0:
            return

        executionTraces[0].cachedStartDecayingBranchTrace = {
            name: SymbolMapper.createSparseTraceArray(scipy.sparse.csc_matrix(value.shape, dtype=numpy.float64))
            for name, value in executionTraces[0].branchTrace.items()
        }

        executionTraces[0].cachedEndDecayingBranchTrace = {
            name: SymbolMapper.createSparseTraceArray(value.minimum(1) * self.config['decaying_branch_trace_scale'])
            for name, value in executionTraces[0].branchTrace.items()
        }

        lastTrace = executionTraces[0]

        for trace in executionTraces[1:]:
            if trace.cachedStartDecayingBranchTrace is None:
                trace.cachedStartDecayingBranchTrace = copy.deepcopy(lastTrace.cachedEndDecayingBranchTrace)
                trace.cachedEndDecayingBranchTrace = copy.deepcopy(lastTrace.cachedEndDecayingBranchTrace)

                for fileName in trace.cachedEndDecayingBranchTrace.keys():
                    trace.cachedEndDecayingBranchTrace[fileName] *= self.config['decaying_branch_trace_decay_rate']

                for fileName in trace.branchTrace.keys():
                    branchesExecuted = SymbolMapper.createSparseTraceArray(trace.branchTrace[fileName].minimum(1) * self.config['decaying_branch_trace_scale'])

                    if fileName in trace.cachedEndDecayingBranchTrace:
                        if trace.branchTrace[fileName].shape[0] == trace.cachedEndDecayingBranchTrace[fileName].shape[0]:
                            trace.cachedEndDecayingBranchTrace[fileName] += branchesExecuted
                    else:
                        trace.cachedEndDecayingBranchTrace[fileName] = branchesExecuted

            lastTrace = trace

    def computeCachedDecayingFutureBranchTrace(self, executionTraces):
        # Create the decaying future execution trace for the prediction algorithm
        # The decaying future execution trace is basically a vector that describes
        # what is going to happen in the future. Its similar to the decaying branch
        # trace that is fed as an input to the algorithm. The difference is this.
        # The decaying branch trace shows what happened in the past, with the lines
        # of code that get executed set to 1 in the vector and then decaying thereafter.
        # The decaying future trace is exactly the same but in reverse - it provides
        # what is going to happen next after this trace. The lines of code which
        # execute immediately next are set to 1, and ones that execute further in the
        # future have some decayed value based on the decay rate. What this does is
        # provide an additional, highly supervised target for a secondary loss function.

        if len(executionTraces) == 0:
            return

        reversedExecutionTraces = list(executionTraces)
        reversedExecutionTraces.reverse()

        reversedExecutionTraces[0].cachedEndDecayingFutureBranchTrace = {
            name: SymbolMapper.createSparseTraceArray(scipy.sparse.csc_matrix(value.shape, dtype=numpy.float64))
            for name, value in reversedExecutionTraces[0].branchTrace.items()
        }

        reversedExecutionTraces[0].cachedStartDecayingFutureBranchTrace = {
            name: SymbolMapper.createSparseTraceArray(value.minimum(1) * self.config['decaying_future_branch_trace_scale'])
            for name, value in reversedExecutionTraces[0].branchTrace.items()
        }

        nextTrace = reversedExecutionTraces[0]

        for trace in reversedExecutionTraces[1:]:
            if trace.cachedStartDecayingFutureBranchTrace is None:
                trace.cachedStartDecayingFutureBranchTrace = copy.deepcopy(nextTrace.cachedStartDecayingFutureBranchTrace)

                trace.cachedEndDecayingFutureBranchTrace = copy.deepcopy(nextTrace.cachedStartDecayingFutureBranchTrace)

                for fileName in trace.cachedStartDecayingFutureBranchTrace.keys():
                    trace.cachedStartDecayingFutureBranchTrace[fileName] *= self.config['decaying_future_execution_trace_decay_rate']

                for fileName in trace.branchTrace.keys():
                    traceNumpyArray = trace.branchTrace[fileName]

                    branchesExecuted = SymbolMapper.createSparseTraceArray(traceNumpyArray.minimum(numpy.ones_like(traceNumpyArray, dtype=numpy.float64, shape=traceNumpyArray.shape)) * self.config['decaying_future_branch_trace_scale'])

                    if fileName in trace.cachedStartDecayingFutureBranchTrace:
                        if trace.branchTrace[fileName].shape[0] == trace.cachedStartDecayingFutureBranchTrace[fileName].shape[0]:
                            trace.cachedStartDecayingFutureBranchTrace[fileName] += branchesExecuted
                    else:
                        trace.cachedStartDecayingFutureBranchTrace[fileName] = branchesExecuted

            nextTrace = trace

    @staticmethod
    def createSparseTraceArray(a):
        return scipy.sparse.csc_matrix(a.reshape([a.shape[0], 1]), shape=[a.shape[0], 1], dtype=numpy.float64)
