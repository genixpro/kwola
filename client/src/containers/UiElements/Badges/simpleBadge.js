import React from 'react';
import Badge from './badge.style';
import IconButton from '../../../components/uielements/iconbutton';
import Icon from '../../../components/uielements/icon/index.js';
import Button from '../../../components/uielements/button';

export default ({ classes }) => {
  return (
    <div>
      <Badge className={classes.badge} badgeContent={4} color="primary">
        <Icon>mail</Icon>
      </Badge>
      <Badge className={classes.badge} badgeContent={10} color="secondary">
        <Icon>mail</Icon>
      </Badge>
      <IconButton>
        <Badge className={classes.badge} badgeContent={4} color="primary">
          <Icon>mail</Icon>
        </Badge>
      </IconButton>

      <Badge className={classes.badge} badgeContent={4} color="primary">
        <Button variant="contained">Button</Button>
      </Badge>
    </div>
  );
};
