import actions from './actions';

const initState = {
    executionSession: null,
    executionTraces: null,
    loading: false,
    error: false
};

export default function reducer(state = initState, action) {
    switch (action.type) {
        case actions.EXECUTION_SESSION_REQUEST:
            return {
                ...state,
                executionSession: null,
                executionTraces: null,
                loading: true,
                error: false
            };

        case actions.EXECUTION_SESSION_SUCCESS_RESULT:
            return {
                ...state,
                executionSession: action.executionSession,
                executionTraces: action.executionTraces,
                loading: false,
                error: false
            };

        case actions.EXECUTION_SESSION_ERROR_RESULT:
            return {
                ...state,
                executionSession: null,
                executionTraces: null,
                loading: false,
                error: true
            };


        default:
            return state;
    }
}
