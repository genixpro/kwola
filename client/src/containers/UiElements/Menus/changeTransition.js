import React from "react";
import Button from "../../../components/uielements/button";
import Menus, { MenuItem } from "../../../components/uielements/menus";


class FadeMenu extends React.Component {
  state = {
    anchorEl: null,
    open: false,
  };

  handleClick = event => {
    this.setState({ open: true, anchorEl: event.currentTarget });
  };

  handleRequestClose = () => {
    this.setState({ open: false });
  };

  render() {
    return (
      <div>
        <Button
          aria-owns={this.state.open ? "fade-menu" : null}
          aria-haspopup="true"
          onClick={this.handleClick}
        >

          Open with fade transition
        </Button>
        <Menus
          id="fade-menu"
          anchorEl={this.state.anchorEl}
          open={this.state.open}
          onClose={this.handleRequestClose}
        >
          <MenuItem onClick={this.handleRequestClose}>Profile</MenuItem>
          <MenuItem onClick={this.handleRequestClose}>My account</MenuItem>
          <MenuItem onClick={this.handleRequestClose}>Logout</MenuItem>
        </Menus>
      </div>
    );
  }
}

export default FadeMenu;
