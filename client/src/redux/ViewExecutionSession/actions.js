const actions = {
    EXECUTION_SESSION_REQUEST: 'EXECUTION_SESSION_REQUEST',
    EXECUTION_SESSION_SUCCESS_RESULT: 'EXECUTION_SESSION_SUCCESS_RESULT',
    EXECUTION_SESSION_ERROR_RESULT: 'EXECUTION_SESSION_ERROR_RESULT',

    requestExecutionSession: (id) => ({
        type: actions.EXECUTION_SESSION_REQUEST,
        _id: id
    }),
    executionSessionSuccess: (executionSession) => ({
        type: actions.EXECUTION_SESSION_SUCCESS_RESULT,
        executionSession,
    }),
    executionSessionError: () => ({
        type: actions.EXECUTION_SESSION_ERROR_RESULT
    }),
};
export default actions;
