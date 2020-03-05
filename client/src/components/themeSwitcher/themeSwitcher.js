import React from 'react';
import { withStyles } from '@material-ui/core/styles';
import IntlMessages from '../../components/utility/intlMessages';
import { SwitcherBtns, Button } from './style';

const ThemeSwitcherBtn = ({ config, changeTheme, selectedId }) => {
  const { id, label, options } = config;
  return (
    <SwitcherBtns>
      <h4>
        <IntlMessages id={label} />
      </h4>
      <div className="themeSwitchBtnWrapper">
        {options.map(option => {
          const { themeName, backgroundColor } = option;
          const onClick = () => {
            changeTheme(id, themeName);
          };
          const customClass = themeName === selectedId ? 'selectedTheme' : '';
          return (
            <Button
              key={themeName}
              onClick={onClick}
              className={customClass}
              style={{ backgroundColor: backgroundColor }}
            >
              &nbsp;
            </Button>
          );
        })}
      </div>
    </SwitcherBtns>
  );
};

export default withStyles({})(ThemeSwitcherBtn);
