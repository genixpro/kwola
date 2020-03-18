import { all } from 'redux-saga/effects';
import authSagas from './auth/saga';
import ListApplicationSagas from './ListApplications/sagas';
import ViewTestingSequenceSagas from './ViewTestingSequence/sagas';
import ViewTrainingSequenceSagas from './ViewTrainingSequence/sagas';
import ViewTrainingStepSagas from './ViewTrainingStep/sagas';
import ViewExecutionSessionSagas from './ViewExecutionSession/sagas';
import ViewExecutionTraceSagas from './ViewExecutionTrace/sagas';

export default function* rootSaga(getState) {
  yield all([
      authSagas(),
      ListApplicationSagas(),
      ViewTestingSequenceSagas(),
      ViewTrainingSequenceSagas(),
      ViewTrainingStepSagas(),
      ViewExecutionSessionSagas(),
      ViewExecutionTraceSagas(),
  ]);
}
