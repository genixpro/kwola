import { call, put, takeEvery, takeLatest } from 'redux-saga/effects'
import axios from "axios";

// worker Saga: will be fired on USER_FETCH_REQUESTED actions
function* fetchTrainingStep(action) {
    try {
        const response = yield axios.get(`/api/training_steps/${action._id}`);

        const executionSessionsResponse = yield axios.get(`/api/execution_sessions`);

        yield put({
            type: "TRAINING_STEP_SUCCESS_RESULT",
            trainingStep: response.data.trainingStep,
            executionSessions: executionSessionsResponse.data.executionSessions
        });

    } catch (e) {
        yield put({type: "TRAINING_STEP_ERROR_RESULT", message: e.message});
    }
}



/*
  Starts fetchUser on each dispatched `USER_FETCH_REQUESTED` action.
  Allows concurrent fetches of user.
*/
function* applicationSaga() {
    yield takeEvery("TRAINING_STEP_REQUEST", fetchTrainingStep);
}

export default applicationSaga;

