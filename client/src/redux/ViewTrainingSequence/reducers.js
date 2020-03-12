import actions from './actions';

const initState = {
    application: null,
    trainingSequences: [],
    trainingSteps: [],
    loading: false,
    error: false
};

export default function reducer(state = initState, action) {
    switch (action.type) {
        case actions.TRAINING_SEQUENCE_REQUEST:
            return {
                ...state,
                trainingSequence: null,
                testingSequences: [],
                trainingSteps: [],
                loading: true,
                error: false
            };

        case actions.TRAINING_SEQUENCE_SUCCESS_RESULT:
            return {
                ...state,
                trainingSequence: action.trainingSequence,
                testingSequences: action.testingSequences,
                trainingSteps: action.trainingSteps,
                loading: false,
                error: false
            };

        case actions.TRAINING_SEQUENCE_ERROR_RESULT:
            return {
                ...state,
                trainingSequence: null,
                testingSequences: [],
                trainingSteps: [],
                loading: false,
                error: true
            };


        default:
            return state;
    }
}
