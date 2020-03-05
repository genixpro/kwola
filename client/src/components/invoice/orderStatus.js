import React, { Component } from 'react';
import classNames from 'classnames';
import { MenuItem, MenuList } from '../uielements/menus';
import Grow from '@material-ui/core/Grow';

import { withStyles } from '@material-ui/core/styles';
// import { Manager, Reference, Popper } from 'react-popper';
import Popper from '@material-ui/core/Popper';
import ClickAwayListener from '@material-ui/core/ClickAwayListener';
import Paper from '../uielements/paper';
import Button from '../uielements/button';

const styles = {
  root: {
    display: 'flex',
  },
  popperClose: {
    pointerEvents: 'none',
  },
};

class MenuListComposition extends Component {
  state = {
    open: false,
    anchorEl: null,
  };

  handleClick = event => {
    const { currentTarget } = event;
    this.setState({
      open: true,
      anchorEl: currentTarget,
    });
  };

  handleClose = () => {
    this.setState({ open: false });
  };

  render() {
    const { classes, value, onChange, orderStatusOptions } = this.props;
    const { anchorEl, open } = this.state;
    const id = open ? 'order-drop-list' : null;
    return (
      <div className={classes.root}>
        <Button
          aria-describedby={id}
          onClick={this.handleClick}
          color="primary"
        >
          {value}
        </Button>

        <Popper
          placement="bottom-end"
          className={classNames({ [classes.popperClose]: !open })}
          id="id"
          open={open}
          anchorEl={anchorEl}
        >
          {({ TransitionProps }) => (
            <ClickAwayListener onClickAway={this.handleClose}>
              <Grow in={open} id="menu-list" {...TransitionProps}>
                <Paper>
                  <MenuList role="menu">
                    {orderStatusOptions.map(option => (
                      <MenuItem
                        key={option}
                        onClick={() => {
                          onChange(option);
                          this.handleClose();
                        }}
                      >
                        {option}
                      </MenuItem>
                    ))}
                  </MenuList>
                </Paper>
              </Grow>
            </ClickAwayListener>
          )}
        </Popper>
      </div>
    );
  }
}

export default withStyles(styles)(MenuListComposition);
