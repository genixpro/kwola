const actions = {
    TRAINING_SEQUENCE_REQUEST: 'TRAINING_SEQUENCE_REQUEST',
    TRAINING_SEQUENCE_SUCCESS_RESULT: 'TRAINING_SEQUENCE_SUCCESS_RESULT',
    TRAINING_SEQUENCE_ERROR_RESULT: 'TRAINING_SEQUENCE_ERROR_RESULT',

    requestTrainingSequence: (id) => ({
        type: actions.TRAINING_SEQUENCE_REQUEST,
        _id: id
    }),
    trainingSequenceSuccess: (trainingSequence) => ({
        type: actions.TRAINING_SEQUENCE_SUCCESS_RESULT,
        trainingSequence,
    }),
    trainingSequenceError: () => ({
        type: actions.TRAINING_SEQUENCE_ERROR_RESULT
    }),
};
export default actions;
