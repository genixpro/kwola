import { call, put, takeEvery, takeLatest } from 'redux-saga/effects'
import axios from "axios";

// worker Saga: will be fired on USER_FETCH_REQUESTED actions
function* fetchTrainingSequence(action) {
    try {
        const response = yield axios.get(`/api/trainng_sequences/${action._id}`);

        yield put({type: "TRAINING_SEQUENCE_SUCCESS_RESULT", trainngSequence: response.data.trainngSequence});

    } catch (e) {
        yield put({type: "TRAINING_SEQUENCE_ERROR_RESULT", message: e.message});
    }
}



/*
  Starts fetchUser on each dispatched `USER_FETCH_REQUESTED` action.
  Allows concurrent fetches of user.
*/
function* applicationSaga() {
    yield takeEvery("TRAINING_SEQUENCE_REQUEST", fetchTrainingSequence);
}

export default applicationSaga;

