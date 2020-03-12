import actions from './actions';

const initState = {
    application: null,
    testingSequences: [],
    loading: false,
    error: false
};

export default function reducer(state = initState, action) {
    switch (action.type) {
        case actions.TESTING_SEQUENCE_REQUEST:
            return {
                ...state,
                testingSequence: null,
                loading: true,
                error: false
            };

        case actions.TESTING_SEQUENCE_SUCCESS_RESULT:
            return {
                ...state,
                testingSequence: action.testingSequence,
                loading: false,
                error: false
            };

        case actions.TESTING_SEQUENCE_ERROR_RESULT:
            return {
                ...state,
                testingSequence: null,
                loading: false,
                error: true
            };


        default:
            return state;
    }
}
