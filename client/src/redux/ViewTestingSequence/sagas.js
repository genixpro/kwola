import { call, put, takeEvery, takeLatest } from 'redux-saga/effects'
import axios from "axios";

// worker Saga: will be fired on USER_FETCH_REQUESTED actions
function* fetchTestingSequence(action) {
    try {
        const response = yield axios.get(`/api/testing_sequences/${action._id}`);

        yield put({type: "TESTING_SEQUENCE_SUCCESS_RESULT", testingSequence: response.data.testingSequence});

    } catch (e) {
        yield put({type: "TESTING_SEQUENCE_ERROR_RESULT", message: e.message});
    }
}



/*
  Starts fetchUser on each dispatched `USER_FETCH_REQUESTED` action.
  Allows concurrent fetches of user.
*/
function* applicationSaga() {
    yield takeEvery("TESTING_SEQUENCE_REQUEST", fetchTestingSequence);
}

export default applicationSaga;

