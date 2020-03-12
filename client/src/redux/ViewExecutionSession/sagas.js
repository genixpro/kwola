import { call, put, takeEvery, takeLatest } from 'redux-saga/effects'
import axios from "axios";

// worker Saga: will be fired on USER_FETCH_REQUESTED actions
function* fetchExecutionSession(action) {
    try {
        const response = yield axios.get(`/api/execution_sessions/${action._id}`);

        const executionTracesResponse = yield axios.get(`/api/execution_traces`, {params: {executionSessionId: action._id}});

        yield put({
            type: "EXECUTION_SESSION_SUCCESS_RESULT",
            executionSession: response.data.executionSession,
            executionTraces: executionTracesResponse.data.executionTraces,
        });

    } catch (e) {
        yield put({type: "EXECUTION_SESSION_ERROR_RESULT", message: e.message});
    }
}



/*
  Starts fetchUser on each dispatched `USER_FETCH_REQUESTED` action.
  Allows concurrent fetches of user.
*/
function* applicationSaga() {
    yield takeEvery("EXECUTION_SESSION_REQUEST", fetchExecutionSession);
}

export default applicationSaga;

