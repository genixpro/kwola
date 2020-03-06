import React from 'react';
import PropTypes from 'prop-types';
import Lists, {
  ListItem,
  ListItemText,
  ListItemAvatar,
} from '../../../components/uielements/lists';
import Avatar from '../../../components/uielements/avatars/';
import Icon from '../../../components/uielements/icon/index.js';

function FolderList(props) {
  const { classes } = props;
  return (
    <div className={classes.root}>
      <Lists>
        <ListItem button>
          <ListItemAvatar>
            <Avatar>
              <Icon>folder</Icon>
            </Avatar>
          </ListItemAvatar>
          <ListItemText primary="Photos" secondary="Jan 9, 2016" />
        </ListItem>
        <ListItem button>
          <ListItemAvatar>
            <Avatar>
              <Icon>folder</Icon>
            </Avatar>
          </ListItemAvatar>
          <ListItemText primary="Work" secondary="Jan 7, 2016" />
        </ListItem>
      </Lists>
    </div>
  );
}

FolderList.propTypes = {
  classes: PropTypes.object.isRequired,
};

export default FolderList;
