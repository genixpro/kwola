import React from 'react';
import PropTypes from 'prop-types';
import Lists, {
  ListItem,
  ListItemText,
  ListItemAvatar,
} from '../../../components/uielements/lists';
import Avatar from '../../../components/uielements/avatars';
import Divider from '../../../components/uielements/dividers';
import Icon from '../../../components/uielements/icon/index.js';
import { Root } from './dividers.style';

function InsetDividers(props) {
  return (
    <Root>
      <Lists>
        <ListItem button>
          <ListItemAvatar>
            <Avatar>
              <Icon>folder</Icon>
            </Avatar>
          </ListItemAvatar>
          <ListItemText primary="Work" secondary="Jan 28, 2014" />
        </ListItem>
        <Divider variant="inset" />
        <ListItem button>
          <ListItemAvatar>
            <Avatar>
              <Icon>image</Icon>
            </Avatar>
          </ListItemAvatar>
          <ListItemText primary="Vacation" secondary="Jan 20, 2014" />
        </ListItem>
      </Lists>
    </Root>
  );
}

InsetDividers.propTypes = {
  classes: PropTypes.object.isRequired,
};

export default InsetDividers;
