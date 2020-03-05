import React, { Component } from 'react';
import { findDOMNode } from 'react-dom';
import { connect } from 'react-redux';
import { createMuiTheme } from '@material-ui/core/styles';
import { ThemeProvider } from '@material-ui/styles';
import SwipeableViews from 'react-swipeable-views';
import Scrollbars from '../../components/utility/customScrollBar';
import Tabs, { Tab } from '../../components/uielements/tabs';
import IntlMessages from '../../components/utility/intlMessages';
import TopbarAddtoCart from './topbarAddtoCart';
import TopbarMessage from './topbarMessage';
import { SidebarContent, Icon, CloseButton } from './sidebarNotification.style';
import themeActions from '../../redux/themeSwitcher/actions';
const { switchActivation } = themeActions;

const theme = createMuiTheme({
  overrides: {
    // MuiTab: {
    //   root: {
    //     minWidth: 'auto !important',
    //     color: '#ffffff',
    //   },
    //   wrapper: {
    //     padding: '6px 5px !important',
    //     fontSize: 13,
    //   },
    // },
    // MuiTabs: {
    //   root: {
    //     backgroundColor: '#3F51B5',
    //     // paddingTop: 18,
    //   },
    // },
  },
});

const demoNotifications = [
  {
    id: 1,
    name: 'David Doe',
    notification:
      'A National Book Award Finalist An Edgar Award Finalist A California Book Award Gold Medal Winner',
  },
  {
    id: 2,
    name: 'Navis Doe',
    notification:
      'A National Book Award Finalist An Edgar Award Finalist A California Book Award Gold Medal Winner',
  },
  {
    id: 3,
    name: 'Emanual Doe',
    notification:
      'A National Book Award Finalist An Edgar Award Finalist A California Book Award Gold Medal Winner',
  },
  {
    id: 4,
    name: 'Dowain Doe',
    notification:
      'A National Book Award Finalist An Edgar Award Finalist A California Book Award Gold Medal Winner',
  },
  {
    id: 5,
    name: 'James Doe',
    notification:
      'A National Book Award Finalist An Edgar Award Finalist A California Book Award Gold Medal Winner',
  },
  {
    id: 6,
    name: 'Levene Doe',
    notification:
      'A National Book Award Finalist An Edgar Award Finalist A California Book Award Gold Medal Winner',
  },
  {
    id: 7,
    name: 'Blake Doe',
    notification:
      'A National Book Award Finalist An Edgar Award Finalist A California Book Award Gold Medal Winner',
  },
  {
    id: 8,
    name: 'Ralph Doe',
    notification:
      'A National Book Award Finalist An Edgar Award Finalist A California Book Award Gold Medal Winner',
  },
];

const TabContainer = ({ children, dir }) => {
  return <div>{children}</div>;
};
class TopbarNotification extends Component {
  state = {
    visible: false,
    anchorEl: null,
    tabValue: 0,
  };
  hide = () => {
    this.setState({ visible: false });
  };
  handleVisibleChange = () => {
    this.setState({
      visible: !this.state.visible,
      anchorEl: findDOMNode(this.button),
    });
  };
  notificationContent = height => (
    <SidebarContent
      className="topbarNotification"
      style={{ height: height - 65 }}
    >
      <div className="dropdownBody">
        <Scrollbars style={{ height: '100%' }}>
          {demoNotifications.map(notification => (
            <a href="#!" className="dropdownListItem" key={notification.id}>
              <h5>{notification.name}</h5>
              <p>{notification.notification}</p>
            </a>
          ))}
        </Scrollbars>
      </div>

      <a href="#!" className="viewAllBtn">
        <IntlMessages id="topbar.viewAll" />
      </a>
    </SidebarContent>
  );
  handleTabChanged = (event, tabValue) => {
    this.setState({ tabValue });
  };
  handleChangeIndex = tabValue => {
    this.setState({ tabValue });
  };
  render() {
    const { locale, url, switchActivation, height } = this.props;
    const propsTopbar = { locale, url };
    return (
      <div>
        <CloseButton onClick={() => switchActivation(false)}>
          <Icon>close</Icon>
        </CloseButton>

        <ThemeProvider theme={theme}>
          <Tabs
            value={this.state.tabValue}
            onChange={this.handleTabChanged}
            indicatorColor="secondary"
            textColor="inherit"
            variant="fullWidth"
            style={{ backgroundColor: '#3F51B5' }}
          >
            <Tab label={<IntlMessages id="sidebar.notification" />} />
            <Tab label={<IntlMessages id="sidebar.message" />} />
            <Tab label={<IntlMessages id="sidebar.cart" />} />
          </Tabs>
        </ThemeProvider>

        <SwipeableViews
          axis={'x'}
          index={this.state.tabValue}
          onChangeIndex={this.handleChangeIndex}
        >
          <TabContainer>{this.notificationContent(height)}</TabContainer>
          {/* <TabContainer>
                  <TopbarMail {...propsTopbar} />
                </TabContainer> */}
          <TabContainer>
            <TopbarMessage {...propsTopbar} />
          </TabContainer>
          <TabContainer>
            <TopbarAddtoCart {...propsTopbar} />
          </TabContainer>
        </SwipeableViews>
      </div>
    );
  }
}

export default connect(
  state => ({
    ...state.App,
    customizedTheme: state.ThemeSwitcher.topbarTheme,
    height: state.App.height,
  }),
  { switchActivation }
)(TopbarNotification);
