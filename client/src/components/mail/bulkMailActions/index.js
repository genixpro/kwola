import React, { Component } from 'react';
import { findDOMNode } from 'react-dom';
import { createMuiTheme } from '@material-ui/core/styles';
import { ThemeProvider } from '@material-ui/styles';
import { tags, tagColor, tagsIcon } from '../mailTags';
import notification from '../../notification';
import Divider from '../../uielements/dividers';
import BulkMailActionWrapper, {
  LeftPart,
  RightPart,
  Icon,
  DoneIcon,
  IconButton,
  Popover,
  Checkbox,
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

const countCheckMailed = checkedMails => {
  let count = 0;
  Object.keys(checkedMails).forEach(key => {
    if (checkedMails[key]) {
      count++;
    }
  });
  return count;
};
const allFilterChecked = filterMails => {
  const checkedMails = {};
  filterMails.forEach(mail => {
    checkedMails[mail.id] = true;
  });
  return checkedMails;
};

export default ({
  bulkActions,
  filterMails,
  checkedMails,
  updateCheckedMail,
}) => {
  const countChecked = countCheckMailed(checkedMails);
  if (countChecked === 0) {
    return <div />;
  }
  const allChecked = countChecked === filterMails.length;
  const action = (actionName, value) => {
    const payload = {
      checkedMails,
      action: actionName,
      value,
    };
    bulkActions(payload);
  };
  return (
    <BulkMailActionWrapper>
      <LeftPart>
        <Checkbox
          checked={allChecked}
          onChange={() => {
            if (allChecked) {
              updateCheckedMail({});
            } else {
              updateCheckedMail(allFilterChecked(filterMails));
            }
          }}
        />
        <span className="selectedNum">{`${countChecked} selected`}</span>
      </LeftPart>

      <RightPart>
        <IconButton
          onClick={event => {
            event.stopPropagation();
            action('delete');
            notification('', `${countChecked} mail deleted`, '', true);
          }}
        >
          <Icon>delete</Icon>
        </IconButton>
        <IconButton
          onClick={event => {
            event.stopPropagation();
            action('done');
            notification('', `${countChecked} marked as done`, '', true);
          }}
        >
          <DoneIcon>done</DoneIcon>
        </IconButton>
        <SelectTagButton action={action} />
      </RightPart>
    </BulkMailActionWrapper>
  );
};
