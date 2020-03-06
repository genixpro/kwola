import React from "react";
import PropTypes from "prop-types";
import Icon from "../../../components/uielements/icon/index.js";
import BottomNavigation, {
  BottomNavigationAction
} from "../../../components/uielements/bottomNavigation";

class LabelBottomNavigation extends React.Component {
  state = {
    value: "recents"
  };

  handleChange = (event, value) => {
    this.setState({ value });
  };

  render() {
    const { classes } = this.props;
    const { value } = this.state;

    return (
      <BottomNavigation
        value={value}
        onChange={this.handleChange}
        className={classes.root}
      >
        <BottomNavigationAction
          label="Recents"
          value="recents"
          icon={<Icon>restore</Icon>}
        />
        <BottomNavigationAction
          label="Favorites"
          value="favorites"
          icon={<Icon>favorites</Icon>}
        />
        <BottomNavigationAction
          label="Nearby"
          value="nearby"
          icon={<Icon>location_on</Icon>}
        />
        <BottomNavigationAction
          label="Folder"
          value="folder"
          icon={<Icon>folder</Icon>}
        />
      </BottomNavigation>
    );
  }
}

LabelBottomNavigation.propTypes = {
  classes: PropTypes.object.isRequired
};

export default LabelBottomNavigation;
