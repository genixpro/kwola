import { call, put, takeEvery, takeLatest } from 'redux-saga/effects'
import axios from "axios";

// worker Saga: will be fired on USER_FETCH_REQUESTED actions
function* fetchApplication(action) {
  try {
    const response = yield axios.get(`/api/application/${action._id}`);
    const testingSequencesResponse = yield axios.get(`/api/testing_sequences`);
    const trainingSequencesResponse = yield axios.get(`/api/training_sequences`);

    yield put({
      type: "APPLICATION_SUCCESS_RESULT",
      application: response.data,
      testingSequences: testingSequencesResponse.data.testingSequences,
      trainingSequences: trainingSequencesResponse.data.trainingSequences
    });

  } catch (e) {
    yield put({type: "APPLICATION_ERROR_RESULT", message: e.message});
  }
}



// worker Saga: will be fired on USER_FETCH_REQUESTED actions
function* launchTestingSequence(action) {
  try {
    const values = action.testingSequence;

    const response = yield axios.post(`/api/application/${action.applicationId}/testing_sequences`, {... values});

    yield put({type: "NEW_TESTING_SEQUENCE_SUCCESS_RESULT", testingSequence: response.data});

  } catch (e) {
    yield put({type: "NEW_TESTING_SEQUENCE_ERROR_RESULT", message: e.message});
  }
}


/*
  Starts fetchUser on each dispatched `USER_FETCH_REQUESTED` action.
  Allows concurrent fetches of user.
*/
function* applicationSaga() {
  yield takeEvery("APPLICATION_REQUEST", fetchApplication);
  yield takeEvery("NEW_TESTING_SEQUENCE_REQUEST", launchTestingSequence);
}

export default applicationSaga;

