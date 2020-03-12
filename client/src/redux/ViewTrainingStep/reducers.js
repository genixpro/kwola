import actions from './actions';

const initState = {
    trainingStep: null,
    executionSessions: [],
    loading: false,
    error: false
};

export default function reducer(state = initState, action) {
    switch (action.type) {
        case actions.TRAINING_STEP_REQUEST:
            return {
                ...state,
                trainingStep: null,
                executionSessions: [],
                loading: true,
                error: false
            };

        case actions.TRAINING_STEP_SUCCESS_RESULT:
            return {
                ...state,
                trainingStep: action.trainingStep,
                executionSessions: action.executionSessions,
                loading: false,
                error: false
            };

        case actions.TRAINING_STEP_ERROR_RESULT:
            return {
                ...state,
                trainingStep: null,
                executionSessions: [],
                loading: false,
                error: true
            };


        default:
            return state;
    }
}
