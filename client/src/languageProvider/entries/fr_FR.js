import appLocaleData from '@formatjs/intl-pluralrules/dist/locale-data/fr';
import saMessages from '../locales/fr_FR.json';

const saLang = {
  messages: {
    ...saMessages,
  },
  locale: 'fr-FR',
  data: appLocaleData,
};
export default saLang;
