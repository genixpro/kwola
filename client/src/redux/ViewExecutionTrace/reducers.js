import actions from './actions';

const initState = {
    executionTrace: null,
    loading: false,
    error: false
};

export default function reducer(state = initState, action) {
    switch (action.type) {
        case actions.EXECUTION_TRACE_REQUEST:
            return {
                ...state,
                executionTrace: null,
                loading: true,
                error: false
            };

        case actions.EXECUTION_TRACE_SUCCESS_RESULT:
            return {
                ...state,
                executionTrace: action.executionTrace,
                loading: false,
                error: false
            };

        case actions.EXECUTION_TRACE_ERROR_RESULT:
            return {
                ...state,
                executionTrace: null,
                loading: false,
                error: true
            };


        default:
            return state;
    }
}
