import React from "react";
import PropTypes from "prop-types";
import Lists, {
  ListItem,
  ListItemIcon,
  ListItemSecondaryAction,
  ListItemText,
  ListSubheader
} from "../../../components/uielements/lists";
import Switch from "../../../components/uielements/switch";
import Icon from "../../../components/uielements/icon/index.js";

class SwitchListSecondary extends React.Component {
  state = {
    checked: ["wifi"]
  };

  handleToggle = value => () => {
    const { checked } = this.state;
    const currentIndex = checked.indexOf(value);
    const newChecked = [...checked];

    if (currentIndex === -1) {
      newChecked.push(value);
    } else {
      newChecked.splice(currentIndex, 1);
    }

    this.setState({
      checked: newChecked
    });
  };

  render() {
    const { classes } = this.props;

    return (
      <div className={classes.root}>
        <Lists subheader={<ListSubheader>Settings</ListSubheader>}>
          <ListItem>
            <ListItemIcon>
              <Icon>wifi</Icon>
            </ListItemIcon>
            <ListItemText primary="Wi-Fi" />
            <ListItemSecondaryAction>
              <Switch
                onChange={this.handleToggle("wifi")}
                checked={this.state.checked.indexOf("wifi") !== -1}
              />
            </ListItemSecondaryAction>
          </ListItem>
          <ListItem>
            <ListItemIcon>
              <Icon>bluetooth</Icon>
            </ListItemIcon>
            <ListItemText primary="Bluetooth" />
            <ListItemSecondaryAction>
              <Switch
                onChange={this.handleToggle("bluetooth")}
                checked={this.state.checked.indexOf("bluetooth") !== -1}
              />
            </ListItemSecondaryAction>
          </ListItem>
        </Lists>
      </div>
    );
  }
}

SwitchListSecondary.propTypes = {
  classes: PropTypes.object.isRequired
};

export default SwitchListSecondary;
