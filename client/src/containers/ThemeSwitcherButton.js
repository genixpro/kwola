import React from 'react';
import { connect } from 'react-redux';
import themeActions from '../redux/themeSwitcher/actions';
import { Button, Icon } from './App/style';

const ThemeSwitcherButton = ({ isActivated, switchActivation }) => {
  const openThemeSwitcher = isActivated === 'themeSwitcher';
  const toggleView = () => {
    const value = openThemeSwitcher ? false : 'themeSwitcher';
    switchActivation(value);
  };
  return (
    <Button
      variant="contained"
      color="primary"
      onClick={toggleView}
      className={openThemeSwitcher ? 'active' : ''}
    >
      <Icon>settings</Icon>
    </Button>
  );
};
const mapStateToProps = state => ({
  isActivated: state.ThemeSwitcher.isActivated,
});
export default connect(
  mapStateToProps,
  themeActions
)(ThemeSwitcherButton);
