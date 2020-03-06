import React from 'react';
import PropTypes from 'prop-types';
import Lists, {
  ListItem,
  ListItemText,
  ListItemIcon,
  ListSubheader,
} from '../../../components/uielements/lists';
import { Root } from './lists.style';
import Collapse from '@material-ui/core/Collapse';
import Icon from '../../../components/uielements/icon/index.js';

class NestedList extends React.Component {
  state = { open: true };

  handleClick = () => {
    this.setState({ open: !this.state.open });
  };

  render() {
    const { classes } = this.props;

    return (
      <Root>
        <Lists subheader={<ListSubheader>Nested List Items</ListSubheader>}>
          <ListItem button>
            <ListItemIcon>
              <Icon>send</Icon>
            </ListItemIcon>
            <ListItemText primary="Sent mail" />
          </ListItem>
          <ListItem button>
            <ListItemIcon>
              <Icon>drafts</Icon>
            </ListItemIcon>
            <ListItemText primary="Drafts" />
          </ListItem>
          <ListItem button onClick={this.handleClick}>
            <ListItemIcon>
              <Icon>inbox</Icon>
            </ListItemIcon>
            <ListItemText primary="Inbox" />
            {this.state.open ? (
              <Icon>expand_less</Icon>
            ) : (
              <Icon>expand_more</Icon>
            )}
          </ListItem>
          <Collapse
            component="li"
            in={this.state.open}
            transitionduration="auto"
            unmountOnExit
          >
            <Lists disablePadding>
              <ListItem button className={classes.nested}>
                <ListItemIcon>
                  <Icon>star</Icon>
                </ListItemIcon>
                <ListItemText primary="Starred" />
              </ListItem>
            </Lists>
          </Collapse>
        </Lists>
      </Root>
    );
  }
}

NestedList.propTypes = {
  classes: PropTypes.object.isRequired,
};

export default NestedList;
