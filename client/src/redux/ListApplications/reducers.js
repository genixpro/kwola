import actions from './actions';

const initState = {
  applications: [],
  loading: false,
  error: false
};

export default function reducer(state = initState, action) {
  switch (action.type) {
    case actions.APPLICATION_LIST_REQUEST:
      return {
        ...state,
        loading: true,
        error: false
      };
    case actions.APPLICATION_LIST_SUCCESS_RESULT:
      return {
        ...state,
        applications: action.applications,
        loading: false,
        error: false
      };
    case actions.APPLICATION_LIST_ERROR_RESULT:
      return {
        ...state,
        applications: [],
        loading: false,
        error: true
      };
    default:
      return state;
  }
}
