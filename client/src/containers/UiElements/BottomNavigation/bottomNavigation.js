import React from "react";
import PropTypes from "prop-types";
import Icon from "../../../components/uielements/icon/index.js";
import BottomNavigation, {
  BottomNavigationAction
} from "../../../components/uielements/bottomNavigation";

class SimpleBottomNavigation extends React.Component {
  state = {
    value: 0
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
        showLabels
        className={classes.root}
      >
        <BottomNavigationAction label="Recents" icon={<Icon>restore</Icon>} />
        <BottomNavigationAction
          label="Favorites"
          icon={<Icon>favorites</Icon>}
        />
        <BottomNavigationAction
          label="Nearby"
          icon={<Icon>location_on</Icon>}
        />
      </BottomNavigation>
    );
  }
}

SimpleBottomNavigation.propTypes = {
  classes: PropTypes.object.isRequired
};

export default SimpleBottomNavigation;
