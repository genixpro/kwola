import actions from './actions';

const initState = {
  application: null,
  testingSequences: [],
  loading: false,
  error: false
};

export default function reducer(state = initState, action) {
  switch (action.type) {
    case actions.APPLICATION_REQUEST:
      return {
        ...state,
        loading: true,
        error: false
      };

    case actions.APPLICATION_SUCCESS_RESULT:
      return {
        ...state,
        application: action.application,
        testingSequences: action.testingSequences,
        loading: false,
        error: false
      };

    case actions.APPLICATION_ERROR_RESULT:
      return {
        ...state,
        application: [],
        loading: false,
        error: true
      };

    case actions.NEW_TESTING_SEQUENCE_REQUEST:
      return {
        ...state,
        loading: true,
        error: false
      };

    case actions.NEW_TESTING_SEQUENCE_SUCCESS_RESULT:
      return {
        ...state,
        testingSequence: action.testingSequence,
        loading: false,
        error: false
      };

    case actions.NEW_TESTING_SEQUENCE_ERROR_RESULT:
      return {
        ...state,
        application: [],
        loading: false,
        error: true
      };

    default:
      return state;
  }
}
