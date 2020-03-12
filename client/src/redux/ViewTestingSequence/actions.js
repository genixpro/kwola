const actions = {
    TESTING_SEQUENCE_REQUEST: 'TESTING_SEQUENCE_REQUEST',
    TESTING_SEQUENCE_SUCCESS_RESULT: 'TESTING_SEQUENCE_SUCCESS_RESULT',
    TESTING_SEQUENCE_ERROR_RESULT: 'TESTING_SEQUENCE_ERROR_RESULT',

    requestTestingSequence: (id) => ({
        type: actions.TESTING_SEQUENCE_REQUEST,
        _id: id
    }),
    testingSequenceSuccess: (testingSequence) => ({
        type: actions.TESTING_SEQUENCE_SUCCESS_RESULT,
        testingSequence
    }),
    testingSequenceError: () => ({
        type: actions.TESTING_SEQUENCE_ERROR_RESULT
    }),
};
export default actions;
