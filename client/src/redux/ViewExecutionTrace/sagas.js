import { call, put, takeEvery, takeLatest } from 'redux-saga/effects'
import axios from "axios";

// worker Saga: will be fired on USER_FETCH_REQUESTED actions
function* fetchExecutionTrace(action) {
    try {
        const response = yield axios.get(`/api/execution_traces/${action._id}`);

        yield put({
            type: "EXECUTION_TRACE_SUCCESS_RESULT",
            executionTrace: response.data.executionTrace
        });

    } catch (e) {
        yield put({type: "EXECUTION_TRACE_ERROR_RESULT", message: e.message});
    }
}



/*
  Starts fetchUser on each dispatched `USER_FETCH_REQUESTED` action.
  Allows concurrent fetches of user.
*/
function* applicationSaga() {
    yield takeEvery("EXECUTION_TRACE_REQUEST", fetchExecutionTrace);
}

export default applicationSaga;

