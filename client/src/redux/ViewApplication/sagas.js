import { call, put, takeEvery, takeLatest } from 'redux-saga/effects'
import axios from "axios";

// worker Saga: will be fired on USER_FETCH_REQUESTED actions
function* fetchApplication(action) {
  try {
    const response = yield axios.get(`/api/application/${action._id}`);
    const otherResponse = yield axios.get(`/api/application/${action._id}/testing_sequences`);

    yield put({type: "APPLICATION_SUCCESS_RESULT", application: response.data, testingSequences: otherResponse.data.testingSequences});

  } catch (e) {
    yield put({type: "APPLICATION_ERROR_RESULT", message: e.message});
  }
}

/*
  Starts fetchUser on each dispatched `USER_FETCH_REQUESTED` action.
  Allows concurrent fetches of user.
*/
function* applicationSaga() {
  yield takeEvery("APPLICATION_REQUEST", fetchApplication);
}

export default applicationSaga;
