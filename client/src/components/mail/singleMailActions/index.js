import React, { Component } from 'react';
import { findDOMNode } from 'react-dom';
import { createMuiTheme } from '@material-ui/core/styles';
import { ThemeProvider } from '@material-ui/styles';
import Popover from '../../uielements/popover';
import notification from '../../notification';
import { tags, tagColor, tagsIcon } from '../mailTags';
import Divider from '../../uielements/dividers';
import SingleMailActions, {
  Icon,
  DoneIcon,
  IconButton,
  List,
  ListItem,
  ListSubheader,
  ListItemText,
  ListLabel,
} from './style';

const theme = createMuiTheme({
  overrides: {
    MuiPopover: {
      paper: {
        minWidth: 250,
        marginTop: 10,
      },
    },
  },
});

class SelectTagButton extends Component {
  state = {
    open: false,
    anchorEl: null,
  };
  toggleState = event => {
    if (event && event.stopPropagation) {
      event.stopPropagation();
    }
    this.setState({
      open: !this.state.open,
      anchorEl: findDOMNode(this.button),
    });
  };
  render() {
    const tagOptions = tags.map(tag => (
      <ListItem
        button
        onClick={event => {
          event.stopPropagation();
          this.setState({ open: false });
          this.props.action('tag', tag);
          notification('', `items moved in ${tag}`, '', true);
        }}
        key={tag}
      >
        <Icon style={{ color: tagColor[tags.findIndex(tags => tags === tag)] }}>
          {tagsIcon[tag]}
        </Icon>
        <ListItemText primary={tag} />
      </ListItem>
    ));
    return (
      <div>
        <ThemeProvider theme={theme}>
          <Popover
            open={this.state.open}
            anchorEl={this.state.anchorEl}
            onClose={this.toggleState}
            anchorOrigin={{
              horizontal: 'right',
              vertical: 'bottom',
            }}
            transformOrigin={{
              horizontal: 'right',
              vertical: 'top',
            }}
          >
            <List className="dropdownList">
              <ListLabel>Move selected to...</ListLabel>
              <Divider />

              <List>
                <ListItem
                  button
                  onClick={event => {
                    event.stopPropagation();
                    this.props.action('report');
                    notification('', `1 reported as spam`, '', true);
                  }}
                >
                  <Icon style={{ color: '#776f6c' }}>report</Icon>
                  <ListItemText primary="spam" />
                </ListItem>
              </List>

              <Divider />

              <List
                subheader={
                  <ListSubheader component="div">
                    Nested List Items
                  </ListSubheader>
                }
              >
                {tagOptions}
              </List>
            </List>
          </Popover>
        </ThemeProvider>
        <IconButton
          onClick={this.toggleState}
          ref={node => {
            this.button = node;
          }}
        >
          <Icon>more_vert</Icon>
        </IconButton>
      </div>
    );
  }
}

export default ({ mail, bulkActions }) => {
  const action = (actionName, value) => {
    const payload = {
      checkedMails: {
        [mail.id]: true,
      },
      action: actionName,
      value,
    };
    bulkActions(payload);
  };
  return (
    <SingleMailActions className="singleMailActions">
      <IconButton
        onClick={event => {
          event.stopPropagation();
          action('delete');
          notification('', `mail deleted`, '', true);
        }}
      >
        <Icon>delete</Icon>
      </IconButton>

      <IconButton
        onClick={event => {
          event.stopPropagation();
          action('done');
          notification('', `marked as done`, '', true);
        }}
      >
        <DoneIcon>done</DoneIcon>
      </IconButton>

      <SelectTagButton action={action} />
    </SingleMailActions>
  );
};
