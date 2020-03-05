import { getDefaultPath } from '../../helpers/urlSync';
import actions, { getView } from './actions';
import { themeConfig } from '../../settings';

const preKeys = getDefaultPath();
const topbarHeight = 65;
const initState = {
  // collapsed: window.innerWidth > 1220 ? false : true,
  collapsed: true,
  current: preKeys,
  showBreadCrumb: themeConfig.showBreadCrumb,
  fixedNavbar: themeConfig.fixedNavbar,
  view: getView(window.innerWidth),
  height: window.innerHeight,
  bredHeight: 0,
  scrollHeight: window.innerHeight - topbarHeight,
  openKeys: preKeys,
};

export default function appReducer(state = initState, action) {
  switch (action.type) {
    case actions.COLLPSE_CHANGE:
      return {
        ...state,
        collapsed: !state.collapsed,
      };
    case actions.COLLPSE_OPEN_DRAWER:
      return {
        ...state,
        openDrawer: !state.openDrawer,
      };
    case actions.TOGLLE_FIXED_NAVBAR:
      return {
        ...state,
        fixedNavbar: !state.fixedNavbar,
      };
    case actions.TOGGLE_ALL:
      if (state.view !== action.view || action.height !== state.height) {
        const height = action.height ? action.height : state.height;
        const bredHeight = state.bredHeight;
        const scrollHeight = height - bredHeight - topbarHeight;
        return {
          ...state,
          collapsed: action.collapsed,
          view: action.view,
          height,
          scrollHeight,
        };
      }
      break;
    case actions.TOGLLE_SHOW_BREADCRUMB:
      return {
        ...state,
        showBreadCrumb: !state.showBreadCrumb,
      };
    case actions.CHANGE_BRED_HEIGHT: {
      const height = state.height;
      const scrollHeight = height - action.bredHeight - topbarHeight;
      return {
        ...state,
        bredHeight: action.bredHeight,
        scrollHeight,
      };
    }
    case actions.CHANGE_OPEN_KEYS:
      return {
        ...state,
        openKeys: action.openKeys,
      };
    case actions.CHANGE_CURRENT:
      const fixedNavbar = state.fixedNavbar;
      const view = state.view;
      let collapsed = state.collapsed;
      if (fixedNavbar && view !== 'DesktopView') {
        collapsed = true;
      }
      return {
        ...state,
        current: action.current,
        collapsed,
      };
    case actions.LOAD_APPLICATION_LIST:
      const newList = action.applications;
      return {
        ...state,
        applications:newList
      };
    default:
      return state;
  }
  return state;
}
