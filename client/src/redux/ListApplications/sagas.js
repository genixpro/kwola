import { call, put, takeEvery, takeLatest } from 'redux-saga/effects'
import axios from "axios";

// worker Saga: will be fired on USER_FETCH_REQUESTED actions
function* fetchApplicationList(action) {
  try {
    const response = yield axios.get("/api/application");

    yield put({type: "APPLICATION_LIST_SUCCESS_RESULT", applications: response.data.applications});

  } catch (e) {
    yield put({type: "APPLICATION_LIST_ERROR_RESULT", message: e.message});
  }
}

/*
  Starts fetchUser on each dispatched `USER_FETCH_REQUESTED` action.
  Allows concurrent fetches of user.
*/
function* applicationListSaga() {
  yield takeEvery("APPLICATION_LIST_REQUEST", fetchApplicationList);
}

export default applicationListSaga;
