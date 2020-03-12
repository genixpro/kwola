import { all } from 'redux-saga/effects';
import authSagas from './auth/saga';
import ListApplicationSagas from './ListApplications/sagas';
import ViewApplicationSagas from './ViewApplication/sagas';
import ViewTestingSequenceSagas from './ViewTestingSequence/sagas';
import ViewTrainingSequenceSagas from './ViewTrainingSequence/sagas';
import ViewTrainingStepSagas from './ViewTrainingStep/sagas';
import ViewExecutionSessionSagas from './ViewExecutionSession/sagas';
import ViewExecutionTraceSagas from './ViewExecutionTrace/sagas';

export default function* rootSaga(getState) {
  yield all([
      authSagas(),
      ListApplicationSagas(),
      ViewApplicationSagas(),
      ViewTestingSequenceSagas(),
      ViewTrainingSequenceSagas(),
      ViewTrainingStepSagas(),
      ViewExecutionSessionSagas(),
      ViewExecutionTraceSagas(),
  ]);
}
