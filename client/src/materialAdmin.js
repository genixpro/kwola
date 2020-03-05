import React, { Suspense } from 'react';
import { Provider } from 'react-redux';
import { IntlProvider } from 'react-intl';
import { ThemeProvider } from 'styled-components';
import { create } from 'jss';
import rtl from 'jss-rtl';
import { StylesProvider, jssPreset } from '@material-ui/styles';
import themes from './settings/themes';
import { themeConfig } from './settings';
import AppLocale from './languageProvider';
import { store, history } from './redux/store';
import Boot from './redux/boot';
import Router from './router';
import Loader from './components/utility/Loader/';

const currentAppLocale = AppLocale.en;

if (!global.__INSERTION_POINT__) {
  global.__INSERTION_POINT__ = true;
  const styleNode = document.createComment('insertion-point-jss');

  if (document.head) {
    document.head.insertBefore(styleNode, document.head.firstChild);
  }
}

const jss = create({
  plugins: [...jssPreset().plugins, rtl()],
  insertionPoint: 'insertion-point-jss',
});

const MetaAdmin = () => {
  return (
    <Suspense fallback={<Loader />}>
      <StylesProvider jss={jss}>
        <IntlProvider
          textComponent="span"
          locale={currentAppLocale.locale}
          messages={currentAppLocale.messages}
        >
          <ThemeProvider theme={themes[themeConfig.theme]}>
            <Provider store={store}>
              <Router history={history} />
            </Provider>
          </ThemeProvider>
        </IntlProvider>
      </StylesProvider>
    </Suspense>
  );
};

Boot()
  .then(() => MetaAdmin())
  .catch(error => console.error(error));

export default MetaAdmin;
