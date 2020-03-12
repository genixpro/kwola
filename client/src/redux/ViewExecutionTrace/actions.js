const actions = {
    EXECUTION_TRACE_REQUEST: 'EXECUTION_TRACE_REQUEST',
    EXECUTION_TRACE_SUCCESS_RESULT: 'EXECUTION_TRACE_SUCCESS_RESULT',
    EXECUTION_TRACE_ERROR_RESULT: 'EXECUTION_TRACE_ERROR_RESULT',

    requestExecutionTrace: (id) => ({
        type: actions.EXECUTION_TRACE_REQUEST,
        _id: id
    }),
    executionTraceSuccess: (executionTrace) => ({
        type: actions.EXECUTION_TRACE_SUCCESS_RESULT,
        executionTrace,
    }),
    executionTraceError: () => ({
        type: actions.EXECUTION_TRACE_ERROR_RESULT
    }),
};
export default actions;
