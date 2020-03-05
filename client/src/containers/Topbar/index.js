import React, { Component } from 'react';
import { connect } from 'react-redux';

import Icon from '../../components/uielements/icon';
import appActions from '../../redux/app/actions';
import themeActions from '../../redux/themeSwitcher/actions';
import { AppHolder, Toolbar, IconButtons, TopbarComponents } from './style';
import TopbarSearch from './topbarSearch';
// import SecondarySidebar from '../SecondarySidebar';
import TopbarNotification from './sidebarNotification';
import TopbarUser from './topbarUser';
const { toggleCollapsed } = appActions;
const { switchActivation } = themeActions;

class Topbar extends Component {
  render() {
    const {
      toggleCollapsed,
      locale,
      url,
      customizedTheme,
      switchActivation,
    } = this.props;
    const propsTopbar = { locale, url };
    return (
      <AppHolder style={{ background: customizedTheme.backgroundColor }}>
        <Toolbar
          style={{
            paddingLeft: '30px',
            minHeight: '64px',
            background: customizedTheme.topbarTheme,
          }}
        >
          <IconButtons
            id="topbarCollapsed"
            aria-label="open drawer"
            onClick={toggleCollapsed}
            className="right"
          >
            <Icon>menu</Icon>
          </IconButtons>

          <TopbarComponents>
            <ul className="topbarItems">
              <li className="topbarSearch">
                <TopbarSearch {...propsTopbar} />
              </li>

              {/* <li className="topbarNotification">
                <TopbarNotification {...propsTopbar} />
              </li> */}

              <li className="topbarNotification">
                <div>
                  <Icon
                    onClick={() => switchActivation('notification')}
                    style={{ matginTop: 5 }}
                  >
                    widgets
                  </Icon>
                  {/* <SecondarySidebar
                    InnerComponent={TopbarNotification}
                    currentActiveKey="notification"
                    {...propsTopbar}
                  /> */}
                </div>
              </li>

              <li className="topbarUser">
                <TopbarUser {...propsTopbar} />
              </li>
            </ul>
          </TopbarComponents>
        </Toolbar>
      </AppHolder>
    );
  }
}

export default connect(
  state => ({
    ...state.App,
    locale: state.LanguageSwitcher.language.locale,
    customizedTheme: state.ThemeSwitcher.topbarTheme,
  }),
  { toggleCollapsed, switchActivation }
)(Topbar);
