import React from 'react';
import PropTypes from 'prop-types';
import Lists, {
  ListItem,
  ListItemIcon,
  ListItemText,
} from '../../../components/uielements/lists';
import { Root, Icon } from './lists.style';
import Divider from '../../../components/uielements/dividers';

function SimpleList(props) {
  return (
    <Root>
      <Lists>
        <ListItem button>
          <ListItemIcon>
            <Icon>inbox</Icon>
          </ListItemIcon>
          <ListItemText primary="Inbox" />
        </ListItem>
        <ListItem button>
          <ListItemIcon>
            <Icon>drafts</Icon>
          </ListItemIcon>
          <ListItemText primary="Drafts" />
        </ListItem>
      </Lists>
      <Divider />
      <Lists>
        <ListItem button>
          <ListItemText primary="Trash" />
        </ListItem>
        <ListItem button component="a" href="#simple-list">
          <ListItemText primary="Spam" />
        </ListItem>
      </Lists>
    </Root>
  );
}

SimpleList.propTypes = {
  classes: PropTypes.object.isRequired,
};

export default SimpleList;
