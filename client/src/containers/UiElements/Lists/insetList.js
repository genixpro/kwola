import React from 'react';
import PropTypes from 'prop-types';
import Lists, {
  ListItem,
  ListItemIcon,
  ListItemText,
} from '../../../components/uielements/lists';
import { Root } from './lists.style';
import Icon from '../../../components/uielements/icon/index.js';

function InsetList(props) {
  return (
    <Root>
      <Lists>
        <ListItem button>
          <ListItemIcon>
            <Icon>star</Icon>
          </ListItemIcon>
          <ListItemText primary="Chelsea Otakan" />
        </ListItem>
        <ListItem button>
          <ListItemText inset primary="Eric Hoffman" />
        </ListItem>
      </Lists>
    </Root>
  );
}

InsetList.propTypes = {
  classes: PropTypes.object.isRequired,
};

export default InsetList;
