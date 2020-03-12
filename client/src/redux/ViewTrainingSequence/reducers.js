import actions from './actions';

const initState = {
    application: null,
    trainngSequences: [],
    loading: false,
    error: false
};

export default function reducer(state = initState, action) {
    switch (action.type) {
        case actions.TRAINING_SEQUENCE_REQUEST:
            return {
                ...state,
                trainngSequence: null,
                loading: true,
                error: false
            };

        case actions.TRAINING_SEQUENCE_SUCCESS_RESULT:
            return {
                ...state,
                trainngSequence: action.trainngSequence,
                loading: false,
                error: false
            };

        case actions.TRAINING_SEQUENCE_ERROR_RESULT:
            return {
                ...state,
                trainngSequence: null,
                loading: false,
                error: true
            };


        default:
            return state;
    }
}
