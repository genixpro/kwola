import React from 'react';
import PropTypes from 'prop-types';
import classNames from 'classnames';
import { Manager, Reference, Popper } from 'react-popper';
import ClickAwayListener from '@material-ui/core/ClickAwayListener';
import Grow from '@material-ui/core/Grow';
import Button from '../../../components/uielements/button';
import { MenuItem, MenuList } from '../../../components/uielements/menus';
import Paper from '../../../components/uielements/paper';

class MenuListComposition extends React.Component {
  state = {
    open: false,
  };

  handleClick = () => {
    this.setState({ open: true });
  };

  handleRequestClose = () => {
    this.setState({ open: false });
  };

  render() {
    const { classes } = this.props;
    const { open } = this.state;

    return (
      <div className={classes.root}>
        <Paper>
          <MenuList>
            <MenuItem>Profile</MenuItem>
            <MenuItem>My account</MenuItem>
            <MenuItem>Logout</MenuItem>
          </MenuList>
        </Paper>
        <Manager>
          <Reference>
            {({ ref }) => (
              <Button
                aria-owns={this.state.open ? 'menu-list' : null}
                aria-haspopup="true"
                onClick={this.handleClick}
              >
                Open Menu
              </Button>
            )}
          </Reference>
          <Popper
            placement="bottom-start"
            eventsEnabled={open}
            className={classNames({ [classes.popperClose]: !open })}
          >
            {({ ref, style, placement, arrowProps }) => (
              <ClickAwayListener onClickAway={this.handleRequestClose}>
                <Grow
                  in={open}
                  id="menu-list"
                  style={{ transformOrigin: '0 0 0' }}
                >
                  <Paper>
                    <MenuList role="menu">
                      <MenuItem onClick={this.handleRequestClose}>
                        Profile
                      </MenuItem>
                      <MenuItem onClick={this.handleRequestClose}>
                        My account
                      </MenuItem>
                      <MenuItem onClick={this.handleRequestClose}>
                        Logout
                      </MenuItem>
                    </MenuList>
                  </Paper>
                </Grow>
              </ClickAwayListener>
            )}
          </Popper>
        </Manager>
      </div>
    );
  }
}

MenuListComposition.propTypes = {
  classes: PropTypes.object.isRequired,
};

export default MenuListComposition;
