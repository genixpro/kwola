import Auth from './auth/reducer';
import App from './app/reducer';
import ThemeSwitcher from './themeSwitcher/reducer';
import LanguageSwitcher from './languageSwitcher/reducer';
import ListApplications from './ListApplications/reducers';
import ViewTestingSequence from './ViewTestingSequence/reducers';
import ViewTrainingSequence from './ViewTrainingSequence/reducers';
import ViewTrainingStep from './ViewTrainingStep/reducers';
import ViewExecutionSession from './ViewExecutionSession/reducers';
import ViewExecutionTrace from './ViewExecutionTrace/reducers';

export default {
  Auth,
  App,
  ThemeSwitcher,
  LanguageSwitcher,
  ListApplications,
  ViewTestingSequence,
  ViewTrainingSequence,
  ViewTrainingStep,
  ViewExecutionSession,
  ViewExecutionTrace
};
