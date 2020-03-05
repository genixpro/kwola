import { themeConfig } from '../../settings';
const changeThemes = {
  id: 'changeThemes',
  label: 'themeSwitcher',
  defaultTheme: themeConfig.theme,
  options: [
    {
      themeName: 'themedefault',
      buttonColor: '#ffffff',
      textColor: '#323332',
    },
    {
      themeName: 'theme2',
      buttonColor: '#ffffff',
      textColor: '#323332',
    },
  ],
};
const topbarTheme = {
  id: 'topbarTheme',
  label: 'themeSwitcher.Topbar',
  defaultTheme: themeConfig.topbar,
  options: [
    {
      themeName: 'themedefault',
      buttonColor: '#ffffff',
      backgroundColor: '#3949AB',
      textColor: '#323332',
    },
    {
      themeName: 'theme1',
      buttonColor: '#E53935',
      backgroundColor: '#E53935',
      textColor: '#ffffff',
    },
    {
      themeName: 'theme2',
      buttonColor: '#5E35B1',
      backgroundColor: '#5E35B1',
      textColor: '#ffffff',
    },
    {
      themeName: 'theme3',
      buttonColor: '#8E24AA',
      backgroundColor: '#8E24AA',
      textColor: '#ffffff',
    },
    {
      themeName: 'theme4',
      buttonColor: '#43A047',
      backgroundColor: '#43A047',
      textColor: '#ffffff',
    },
    {
      themeName: 'theme5',
      buttonColor: '#D81B60',
      backgroundColor: '#D81B60',
      textColor: '#ffffff',
    },
    {
      themeName: 'theme6',
      buttonColor: '#ffffff',
      backgroundColor: '#1E88E5',
      textColor: '#ffffff',
    },
    {
      themeName: 'theme7',
      buttonColor: '#546E7A',
      backgroundColor: '#546E7A',
      textColor: '#ffffff',
    },
  ],
};
const sidebarTheme = {
  id: 'sidebarTheme',
  label: 'themeSwitcher.Sidebar',
  defaultTheme: themeConfig.sidebar,
  options: [
    {
      themeName: 'themedefault',
      buttonColor: '#2B2B2B',
      backgroundColor: '#2B2B2B',
      textColor: '#ffffff',
    },
    {
      themeName: 'theme1',
      buttonColor: '#E53935',
      backgroundColor: '#E53935',
      textColor: '#ffffff',
    },
    {
      themeName: 'theme2',
      buttonColor: '#5E35B1',
      backgroundColor: '#5E35B1',
      textColor: '#ffffff',
    },
    {
      themeName: 'theme3',
      buttonColor: '#8E24AA',
      backgroundColor: '#8E24AA',
      textColor: '#ffffff',
    },
    {
      themeName: 'theme4',
      buttonColor: '#43A047',
      backgroundColor: '#43A047',
      textColor: '#ffffff',
    },
    {
      themeName: 'theme5',
      buttonColor: '#D81B60',
      backgroundColor: '#D81B60',
      textColor: '#ffffff',
    },
    {
      themeName: 'theme6',
      buttonColor: '#3949AB',
      backgroundColor: '#3949AB',
      textColor: '#ffffff',
    },
    {
      themeName: 'theme7',
      buttonColor: '#546E7A',
      backgroundColor: '#546E7A',
      textColor: '#ffffff',
    },
  ],
};
const layoutTheme = {
  id: 'layoutTheme',
  label: 'themeSwitcher.Background',
  defaultTheme: themeConfig.layout,
  options: [
    {
      themeName: 'themedefault',
      buttonColor: '#ffffff',
      backgroundColor: '#F1F3F6',
      textColor: undefined,
    },
    {
      themeName: 'theme1',
      buttonColor: '#ffffff',
      backgroundColor: '#ffffff',
      textColor: '#323232',
    },
    {
      themeName: 'theme2',
      buttonColor: '#F9F9F9',
      backgroundColor: '#F9F9F9',
      textColor: '#ffffff',
    },
    {
      themeName: 'theme3',
      buttonColor: '#ebebeb',
      backgroundColor: '#ebebeb',
      textColor: '#ffffff',
    },
  ],
};
const breadCrumbTheme = {
  id: 'breadCrumbTheme',
  label: 'themeSwitcher.BreadCrumb',
  defaultTheme: themeConfig.breadCrumbTheme,
  options: [
    {
      themeName: 'themedefault',
      backgroundColor: '#5C6BC0',
      textColor: '#ffffff',
    },
    {
      themeName: 'theme1',
      backgroundColor: '#EF5350',
      textColor: '#ffffff',
    },
    {
      themeName: 'theme2',
      backgroundColor: '#7E57C2',
      textColor: '#ffffff',
    },
    {
      themeName: 'theme3',
      backgroundColor: '#AB47BC',
      textColor: '#ffffff',
    },
    {
      themeName: 'theme4',
      backgroundColor: '#66BB6A',
      textColor: '#ffffff',
    },
    {
      themeName: 'theme5',
      backgroundColor: '#EC407A',
      textColor: '#ffffff',
    },
    {
      themeName: 'theme6',
      backgroundColor: '#42A5F5',
      textColor: '#ffffff',
    },
    {
      themeName: 'theme7',
      backgroundColor: '#78909C',
      textColor: '#ffffff',
    },
  ],
};
const customizedThemes = {
  changeThemes,
  topbarTheme,
  sidebarTheme,
  layoutTheme,
  breadCrumbTheme,
};
export function getCurrentTheme(attribute, selectedThemename) {
  let selecetedTheme = {};
  customizedThemes[attribute].options.forEach(theme => {
    if (theme.themeName === selectedThemename) {
      selecetedTheme = theme;
    }
  });
  return selecetedTheme;
}
export default customizedThemes;
