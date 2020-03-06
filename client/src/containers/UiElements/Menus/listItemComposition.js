import React from "react";
import PropTypes from "prop-types";
import { MenuList, MenuItem } from "../../../components/uielements/menus";
import Paper from "../../../components/uielements/paper";
import Icon from "../../../components/uielements/icon/index.js";
import {
  ListItemText,
  ListItemIcon
} from "../../../components/uielements/lists";

function ListItemComposition(props) {
  const { classes } = props;

  return (
    <Paper>
      <MenuList>
        <MenuItem className={classes.menuItem}>
          <ListItemIcon className={classes.icon}>
            <Icon>send</Icon>
          </ListItemIcon>
          <ListItemText
            classes={{ primary: classes.primary }}
            inset
            primary="Sent mail"
          />
        </MenuItem>
        <MenuItem className={classes.menuItem}>
          <ListItemIcon className={classes.icon}>
            <Icon>drafts</Icon>
          </ListItemIcon>
          <ListItemText
            classes={{ primary: classes.primary }}
            inset
            primary="Drafts"
          />
        </MenuItem>
        <MenuItem className={classes.menuItem}>
          <ListItemIcon className={classes.icon}>
            <Icon>inbox</Icon>
          </ListItemIcon>
          <ListItemText
            classes={{ primary: classes.primary }}
            inset
            primary="Inbox"
          />
        </MenuItem>
      </MenuList>
    </Paper>
  );
}

ListItemComposition.propTypes = {
  classes: PropTypes.object.isRequired
};

export default ListItemComposition;
