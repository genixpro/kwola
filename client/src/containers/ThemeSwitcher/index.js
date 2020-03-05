import React, { Component } from 'react';
import { connect } from 'react-redux';
import IntlMessages from '../../components/utility/intlMessages';
import Switch from '../../components/uielements/switch';
import Actions from '../../redux/themeSwitcher/actions.js';
import appActions from '../../redux/app/actions.js';
import Switcher from '../../components/themeSwitcher/themeSwitcher';
import LanguageSwitcher from '../LanguageSwitcher';
import Themes from './config';
import {
  Icon,
  CloseButton,
  BreadCrumbSwitch,
  ThemeSwitcherHeader,
  SwitcherBlock,
  PurchaseActionBtn,
  SidebarInner,
} from './themeSwitcher.style';

class ThemeSwitcher extends Component {
  render() {
    const {
      topbarTheme,
      sidebarTheme,
      breadCrumbTheme,
      switchActivation,
      changeTheme,
      showBreadCrumb,
      toggleShowBreadCrumb,
      fixedNavbar,
      toggleFixedNavbar,
    } = this.props;
    return (
      <SidebarInner>
        <ThemeSwitcherHeader>
          <h3 className="componentTitle">
            <IntlMessages id="themeSwitcher.settings" />
          </h3>
        </ThemeSwitcherHeader>
        <CloseButton onClick={() => switchActivation(false)}>
          <Icon>close</Icon>
        </CloseButton>
        <SwitcherBlock>
          <BreadCrumbSwitch>
            <h3>
              <IntlMessages id="themeSwitcher.showBreadCrumb" />
            </h3>
            <Switch checked={showBreadCrumb} onChange={toggleShowBreadCrumb} />
          </BreadCrumbSwitch>
          <BreadCrumbSwitch>
            <h3>
              <IntlMessages id="themeSwitcher.fixedNavbar" />
            </h3>
            <Switch checked={fixedNavbar} onChange={toggleFixedNavbar} />
          </BreadCrumbSwitch>

          <Switcher
            config={Themes.topbarTheme}
            changeTheme={changeTheme}
            selectedId={topbarTheme.themeName}
          />

          <Switcher
            config={Themes.breadCrumbTheme}
            changeTheme={changeTheme}
            selectedId={breadCrumbTheme.themeName}
          />
          <Switcher
            config={Themes.sidebarTheme}
            changeTheme={changeTheme}
            selectedId={sidebarTheme.themeName}
          />
          <LanguageSwitcher />
        </SwitcherBlock>

        <PurchaseActionBtn>
          <a href="#!" className="purchaseBtn">
            <IntlMessages id="themeSwitcher.purchase" />
          </a>
        </PurchaseActionBtn>
      </SidebarInner>
    );
  }
}
function mapStateToProps(state) {
  return {
    ...state.ThemeSwitcher,
    ...state.App,
    LanguageSwitcher: state.LanguageSwitcher,
  };
}
export default connect(
  mapStateToProps,
  { ...Actions, ...appActions }
)(ThemeSwitcher);
