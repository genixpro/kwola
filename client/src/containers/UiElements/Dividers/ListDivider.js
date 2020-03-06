import React from 'react';
import PropTypes from 'prop-types';
import Lists, {
  ListItem,
  ListItemText,
} from '../../../components/uielements/lists';
import Divider from '../../../components/uielements/dividers';
import { Root } from './dividers.style';

function ListDividers(props) {
  return (
    <Root>
      <Lists>
        <ListItem button>
          <ListItemText primary="Inbox" />
        </ListItem>
        <Divider light />
        <ListItem button>
          <ListItemText primary="Drafts" />
        </ListItem>
        <Divider />
        <ListItem button>
          <ListItemText primary="Trash" />
        </ListItem>
      </Lists>
    </Root>
  );
}

ListDividers.propTypes = {
  classes: PropTypes.object.isRequired,
};

export default ListDividers;
