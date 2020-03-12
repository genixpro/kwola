import { call, put, takeEvery, takeLatest } from 'redux-saga/effects'
import axios from "axios";

// worker Saga: will be fired on USER_FETCH_REQUESTED actions
function* fetchTrainingSequence(action) {
    try {
        const response = yield axios.get(`/api/training_sequences/${action._id}`);

        const testingSequencesResponse = yield axios.get(`/api/testing_sequences`);

        const trainingStepsResponse = yield axios.get(`/api/training_steps`);

        yield put({
            type: "TRAINING_SEQUENCE_SUCCESS_RESULT",
            trainingSequence: response.data.trainingSequence,
            testingSequences: testingSequencesResponse.data.testingSequences,
            trainingSteps: trainingStepsResponse.data.trainingSteps
        });

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

