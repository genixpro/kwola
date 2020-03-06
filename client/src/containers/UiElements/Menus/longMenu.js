import React from "react";
import IconButton from "../../../components/uielements/iconbutton";
import Icon from "../../../components/uielements/icon/index.js";
import Menus, { MenuItem } from "../../../components/uielements/menus";

const options = [
  "None",
  "Atria",
  "Callisto",
  "Dione",
  "Ganymede",
  "Hangouts Call",
  "Luna",
  "Oberon",
  "Phobos",
  "Pyxis",
  "Sedna",
  "Titania",
  "Triton",
  "Umbriel"
];

const ITEM_HEIGHT = 48;

class LongMenu extends React.Component {
  state = {
    anchorEl: null
  };

  handleClick = event => {
    this.setState({ anchorEl: event.currentTarget });
  };

  handleRequestClose = () => {
    this.setState({ anchorEl: null });
  };

  render() {
    const open = Boolean(this.state.anchorEl);

    return (
      <div>
        <IconButton
          aria-label="More"
          aria-owns={open ? "long-menu" : null}
          aria-haspopup="true"
          onClick={this.handleClick}
        >
          <Icon>more_vert</Icon>
        </IconButton>
        <Menus
          id="long-menu"
          anchorEl={this.state.anchorEl}
          open={open}
          onClose={this.handleRequestClose}
          PaperProps={{
            style: {
              maxHeight: ITEM_HEIGHT * 4.5,
              width: 200
            }
          }}
        >
          {options.map(option => (
            <MenuItem
              key={option}
              selected={option === "Pyxis"}
              onClick={this.handleRequestClose}
            >
              {option}
            </MenuItem>
          ))}
        </Menus>
      </div>
    );
  }
}

export default LongMenu;
