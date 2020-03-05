import React, { Component } from 'react';
import { connect } from 'react-redux';
import { Link } from 'react-router-dom';
import Collapse from '@material-ui/core/Collapse';
import Scrollbars from '../../components/utility/customScrollBar';
import IntlMessages from '../../components/utility/intlMessages';
import appActions from '../../redux/app/actions';
// import Logo from '../../images/logo.png';
import options from './options';
import Drawer, {
  LogoWrapper,
  Lists,
  ListItem,
  ListItemIcon,
  ListItemText,
  ExpandLessIcon,
  ExpandMoreIcon,
} from './style';

const { toggleCollapsed, changeOpenKeys, changeCurrent } = appActions;

let selectedTheme = {};

const ListLabel = ({ label }) => (
  <ListItemText
    style={{ color: selectedTheme.textColor }}
    inset
    primary={<IntlMessages id={label} />}
  />
);
const ListElement = ({
  leftIcon,
  label,
  children,
  optionCollapsed,
  isOpened,
  isNavTab,
}) => {
  return (
    <div className="ListItemWrapper">
      {leftIcon ? (
        <ListItemIcon style={{ color: selectedTheme.textColor }}>
          {leftIcon}
        </ListItemIcon>
      ) : null}
      <ListLabel label={label} />
      {children && !isNavTab ? (
        optionCollapsed ? (
          <ExpandLessIcon style={{ color: selectedTheme.textColor }}>
            expand_less
          </ExpandLessIcon>
        ) : (
          <ExpandMoreIcon style={{ color: selectedTheme.textColor }}>
            expand_more
          </ExpandMoreIcon>
        )
      ) : (
        ''
      )}
    </div>
  );
};

const LogoElem = ({ onLogo }) => {
  return (
    <Link to="/dashboard" onClick={onLogo}>
      {/* <img src={Logo} alt="Logo" /> */}
      Kwola
    </Link>
  );
};

const stripTrailingSlash = str => {
  if (str.substr(-1) === '/') {
    return str.substr(0, str.length - 1);
  }
  return str;
};
class Sidebar extends Component {
  handleClick = () => {};
  onLogo = () => {
    const { changeOpenKeys, changeCurrent, toggleCollapsed } = this.props;
    changeOpenKeys({});
    changeCurrent({});
    toggleCollapsed();
  };
  render() {
    const {
      changeOpenKeys,
      openKeys,
      collapsed,
      changeCurrent,
      current,
      anchor,
      height,
      customizedTheme,
      toggleCollapsed,
      fixedNavbar,
      view,
    } = this.props;
    selectedTheme = customizedTheme;
    const scrollheight = height;
    const url = stripTrailingSlash(this.props.url);
    const menuItem = option => {
      const { key, children, isNavTab } = option;
      const optionCollapsed = children && openKeys[key] === true;
      const isOpened = openKeys[key] === true;
      const linkTo = option.withoutDashboard ? `/${key}` : `${url}/${key}`;
      const collapsedClick = () => {
        if (children && !isNavTab) {
          changeOpenKeys({ [key]: !optionCollapsed });
        } else {
          changeCurrent({ [key]: !optionCollapsed });
        }
      };
      return (
        <div key={key}>
          {children && !isNavTab ? (
            <ListItem
              button
              onClick={collapsedClick}
              className={optionCollapsed ? 'expands' : ''}
            >
              <ListElement
                {...option}
                isOpened={isOpened}
                optionCollapsed={optionCollapsed}
              />
            </ListItem>
          ) : (
            <ListItem
              className={current[key] ? 'selected' : ''}
              onClick={collapsedClick}
            >
              <Link to={linkTo} onClick={toggleCollapsed}>
                <ListElement
                  {...option}
                  isOpened={isOpened}
                  optionCollapsed={optionCollapsed}
                />
              </Link>
            </ListItem>
          )}
          {optionCollapsed && !isNavTab ? (
            <Collapse in={true} timeout={200} unmountOnExit>
              {children.map(menuItem)}
            </Collapse>
          ) : (
            ''
          )}
        </div>
      );
    };
    return (
      <Drawer
        variant={
          view !== 'TabLandView' && view !== 'DesktopView'
            ? undefined
            : fixedNavbar
              ? 'permanent'
              : undefined
        }
        open={!collapsed}
        onClose={toggleCollapsed}
        anchor={anchor}
        transitionDuration={100}
        className={`${collapsed && 'collapsed'} ${fixedNavbar && 'f1x3dnAV'}`}
      >
        <div
          className="drawerInner"
          style={{ background: customizedTheme.backgroundColor }}
        >
          <LogoWrapper>
            <LogoElem onLogo={this.onLogo} />
          </LogoWrapper>
          <Scrollbars style={{ height: scrollheight - 64 }}>
            <Lists>{options.map(menuItem)}</Lists>
          </Scrollbars>
        </div>
      </Drawer>
    );
  }
}

export default connect(
  state => ({
    ...state.App,
    customizedTheme: state.ThemeSwitcher.sidebarTheme,
  }),
  { toggleCollapsed, changeOpenKeys, changeCurrent }
)(Sidebar);
