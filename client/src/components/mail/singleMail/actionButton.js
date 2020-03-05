import React, { Component } from 'react';
import { findDOMNode } from 'react-dom';
import { createMuiTheme } from '@material-ui/core/styles';
import { ThemeProvider } from '@material-ui/styles';
import notification from '../../notification';
import Popover from '../../uielements/popover';
import Divider from '../../uielements/dividers';
import {
  Icon,
  IconButton,
  List,
  ListItem,
  ListItemText,
} from './singleMail.style';

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

export default class ActionButton extends Component {
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
    return (
      <div>
        <ThemeProvider theme={theme}>
          {this.props.draftButton ? (
            <Popover
              open={this.state.open}
              anchorEl={this.state.anchorEl}
              onClose={this.toggleState}
              anchorOrigin={{
                horizontal: 'left',
                vertical: 'bottom',
              }}
              transformOrigin={{
                horizontal: 'left',
                vertical: 'top',
              }}
            >
              <List className="mAil-dRopd0Wn">
                <List>
                  <ListItem
                    button
                    onClick={event => {
                      event.stopPropagation();
                      this.props.action('report');
                      notification('', `1 reported as spam`, '', true);
                    }}
                  >
                    <Icon style={{ color: '#776f6c' }}>reply</Icon>
                    <ListItemText primary="Reply" />
                  </ListItem>

                  <ListItem
                    button
                    onClick={event => {
                      event.stopPropagation();
                      this.props.action('report');
                      notification('', `1 reported as spam`, '', true);
                    }}
                  >
                    <Icon style={{ color: '#776f6c' }}>forward</Icon>
                    <ListItemText primary="Forward" />
                  </ListItem>
                </List>

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
                    <Icon style={{ color: '#776f6c' }}>print</Icon>
                    <ListItemText primary="Print" />
                  </ListItem>

                  <ListItem
                    button
                    onClick={event => {
                      event.stopPropagation();
                      this.props.action('report');
                      notification('', `1 reported as spam`, '', true);
                    }}
                  >
                    <Icon style={{ color: '#776f6c' }}>code</Icon>
                    <ListItemText primary="Show Orginal" />
                  </ListItem>
                </List>
              </List>
            </Popover>
          ) : (
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
              <List className="mAil-dRopd0Wn">
                <List>
                  <ListItem
                    button
                    onClick={event => {
                      event.stopPropagation();
                      this.props.action('report');
                      notification('', `1 reported as spam`, '', true);
                    }}
                  >
                    <Icon style={{ color: '#776f6c' }}>reply</Icon>
                    <ListItemText primary="Reply" />
                  </ListItem>

                  <ListItem
                    button
                    onClick={event => {
                      event.stopPropagation();
                      this.props.action('report');
                      notification('', `1 reported as spam`, '', true);
                    }}
                  >
                    <Icon style={{ color: '#776f6c' }}>forward</Icon>
                    <ListItemText primary="Forward" />
                  </ListItem>
                </List>

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
                    <Icon style={{ color: '#776f6c' }}>print</Icon>
                    <ListItemText primary="Print" />
                  </ListItem>

                  <ListItem
                    button
                    onClick={event => {
                      event.stopPropagation();
                      this.props.action('report');
                      notification('', `1 reported as spam`, '', true);
                    }}
                  >
                    <Icon style={{ color: '#776f6c' }}>code</Icon>
                    <ListItemText primary="Show Orginal" />
                  </ListItem>
                </List>
              </List>
            </Popover>
          )}
        </ThemeProvider>

        {this.props.draftButton ? (
          <p
            className="draftLabel"
            onClick={this.toggleState}
            ref={node => {
              this.button = node;
            }}
          >
            <span>Draft</span> to me <Icon>arrow_drop_down</Icon>
          </p>
        ) : (
          <IconButton
            onClick={this.toggleState}
            ref={node => {
              this.button = node;
            }}
          >
            <Icon>more_vert</Icon>
          </IconButton>
        )}
      </div>
    );
  }
}
