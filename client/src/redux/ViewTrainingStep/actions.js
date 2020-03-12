const actions = {
    TRAINING_STEP_REQUEST: 'TRAINING_STEP_REQUEST',
    TRAINING_STEP_SUCCESS_RESULT: 'TRAINING_STEP_SUCCESS_RESULT',
    TRAINING_STEP_ERROR_RESULT: 'TRAINING_STEP_ERROR_RESULT',

    requestTrainingStep: (id) => ({
        type: actions.TRAINING_STEP_REQUEST,
        _id: id
    }),
    trainingStepSuccess: (trainingStep) => ({
        type: actions.TRAINING_STEP_SUCCESS_RESULT,
        trainingStep,
    }),
    trainingStepError: () => ({
        type: actions.TRAINING_STEP_ERROR_RESULT
    }),
};
export default actions;
