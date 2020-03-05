import clone from 'clone';
import config, { getCurrentTheme } from '../../containers/ThemeSwitcher/config';
import actions from './actions';

const initState = {
  isActivated: false,
  changeThemes: getCurrentTheme(
    'changeThemes',
    config.changeThemes.defaultTheme || 'themedefault'
  ),
  breadCrumbTheme: getCurrentTheme(
    'breadCrumbTheme',
    config.breadCrumbTheme.defaultTheme || 'themedefault'
  ),
  topbarTheme: getCurrentTheme(
    'topbarTheme',
    config.topbarTheme.defaultTheme || 'themedefault'
  ),
  sidebarTheme: getCurrentTheme(
    'sidebarTheme',
    config.sidebarTheme.defaultTheme || 'themedefault'
  ),
  layoutTheme: getCurrentTheme(
    'layoutTheme',
    config.layoutTheme.defaultTheme || 'themedefault'
  )
};

export default function(state = initState, action) {
  switch (action.type) {
    case actions.SWITCH_ACTIVATION:
      return { ...state, isActivated: clone(action.isActivated) };
    case actions.CHANGE_THEME:
      return { ...state, [action.attribute]: action.theme };
    default:
      return state;
  }
}
