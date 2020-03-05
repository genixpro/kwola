import React, { Component } from 'react';
import { connect } from 'react-redux';
import { Link } from 'react-router-dom';
import { getBreadcrumbOption } from '../Sidebar/options';
import IntlMessages from '../../components/utility/intlMessages';
import appActions from '../../redux/app/actions';
import BreadcrumbWrapper, {
  NavRoutes,
  NavLinks,
  PageInfo,
  FakeBoxWithTab,
  FakeBox,
  // Icons,
} from './style';

const bredContainerId = 'bredContainer';
const EmptyBreadcrumb = () => <div id={bredContainerId} />;

class PageBreadcrumb extends Component {
  state = {
    selectedNav: null,
  };
  onHomeCLick = () => {
    const { changeOpenKeys, changeCurrent } = this.props;
    changeOpenKeys({});
    changeCurrent({});
  };
  componentDidMount() {
    this.onBredHeightChange();
  }
  componentDidUpdate() {
    this.onBredHeightChange();
  }
  onBredHeightChange = () => {
    let height = 0;
    try {
      height = document.getElementById(bredContainerId).clientHeight;
    } catch (e) {}
    const { bredHeight, changeBredHeight } = this.props;
    if (height !== bredHeight) {
      changeBredHeight(height);
    }
  };
  render() {
    const { url, showBreadCrumb, customizedTheme, style } = this.props;
    const { parent, activeChildren } = getBreadcrumbOption();
    if (!showBreadCrumb || !parent) {
      return <EmptyBreadcrumb />;
    }
    // const { label, key, icon, hideBreadCrumb } = activeChildren || parent;
    const { label, key, hideBreadCrumb } = activeChildren || parent;
    if (hideBreadCrumb) {
      return <EmptyBreadcrumb />;
    }
    const isNavTab = parent ? parent.isNavTab : null;
    // const LeftIcon = icon ? icon : parent.leftIcon ? parent.leftIcon : '';

    const navLinksOptions = option => (
      <Link
        key={option.key}
        to={`${url}/${option.key}`}
        className={option.key === key ? 'active' : ''}
        onClick={() => this.setState({ selectedNav: option.key })}
      >
        <IntlMessages id={option.label} />
      </Link>
    );
    return (
      <div style={style} id={bredContainerId}>
        {isNavTab ? <FakeBoxWithTab /> : <FakeBox />}
        <BreadcrumbWrapper
          style={{ background: customizedTheme.backgroundColor }}
        >
          <PageInfo>
            {/* <Icons>{LeftIcon}</Icons> */}
            <h3 className="pageTitle">
              <IntlMessages id={label} />
            </h3>
          </PageInfo>

          <NavRoutes>
            <Link to={`${url}`} onClick={this.onHomeCLick}>
              {url.replace('/', '')}
            </Link>
            <span className="currentPageName">{key}</span>
          </NavRoutes>

          {isNavTab ? (
            <NavLinks>{parent.children.map(navLinksOptions)}</NavLinks>
          ) : (
            ''
          )}
        </BreadcrumbWrapper>
      </div>
    );
  }
}
const mapStateToProps = state => {
  return {
    ...state.App,
    customizedTheme: state.ThemeSwitcher.breadCrumbTheme,
  };
};
export default connect(mapStateToProps, appActions)(PageBreadcrumb);
