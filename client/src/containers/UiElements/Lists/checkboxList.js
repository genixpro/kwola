import React from 'react';
import PropTypes from 'prop-types';
import Lists, {
  ListItem,
  ListItemText,
  ListItemIcon,
  ListItemSecondaryAction,
} from '../../../components/uielements/lists';
import Checkbox from '../../../components/uielements/checkbox/';
import IconButton from '../../../components/uielements/iconbutton/';
import Icon from '../../../components/uielements/icon/index.js';

class CheckboxList extends React.Component {
  state = {
    checked: [0],
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
      checked: newChecked,
    });
  };

  render() {
    const { classes } = this.props;

    return (
      <div className={classes.root}>
        <Lists>
          {[0, 1, 2, 3].map(value => (
            <ListItem
              key={value}
              dense
              button
              onClick={this.handleToggle(value)}
              className={classes.listItem}
            >
              <ListItemIcon>
                <Checkbox
                  checked={this.state.checked.indexOf(value) !== -1}
                  tabIndex={-1}
                  disableRipple
                />
              </ListItemIcon>
              <ListItemText primary={`Line item ${value + 1}`} />
              <ListItemSecondaryAction>
                <IconButton aria-label="Comments">
                  <Icon>comment</Icon>
                </IconButton>
              </ListItemSecondaryAction>
            </ListItem>
          ))}
        </Lists>
      </div>
    );
  }
}

CheckboxList.propTypes = {
  classes: PropTypes.object.isRequired,
};

export default CheckboxList;
