import React from 'react';
import PropTypes from 'prop-types';
import Lists, {
  ListItem,
  ListItemText,
  ListItemAvatar,
  ListItemSecondaryAction,
} from '../../../components/uielements/lists';
import { Root } from './lists.style';
import Checkbox from '../../../components/uielements/checkbox/';
import Avatar from '../../../components/uielements/avatars/';
import legend from '../../../images/user.jpg';

class CheckboxListSecondary extends React.Component {
  state = {
    checked: [1],
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
      <Root>
        <Lists>
          {[1, 2, 4].map(value => (
            <ListItem key={value} dense button className={classes.listItem}>
              <ListItemAvatar>
                <Avatar alt="Remy Sharp" src={legend} />
              </ListItemAvatar>
              <ListItemText primary={`Position ${value + 1}`} />
              <ListItemSecondaryAction>
                <Checkbox
                  onChange={this.handleToggle(value)}
                  checked={this.state.checked.indexOf(value) !== -1}
                />
              </ListItemSecondaryAction>
            </ListItem>
          ))}
        </Lists>
      </Root>
    );
  }
}

CheckboxListSecondary.propTypes = {
  classes: PropTypes.object.isRequired,
};

export default CheckboxListSecondary;
