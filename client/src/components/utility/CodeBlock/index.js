import React, { Component } from 'react';
import { createMuiTheme } from '@material-ui/core/styles';
import { ThemeProvider } from '@material-ui/styles';
import ExpansionPanel, {
  Title,
  ExpansionPanelSummary,
  ExpansionPanelDetails,
  CodeMirror,
  ExpandIcon,
  Button,
  Icon,
  ExpansionPanelDetailsWrapper,
} from './codeBlock.style';
import { Fab } from '../../uielements/button';
import CopyToClipboard from '../../uielements/copyToClipboard';
import notification from '../../notification';
import getPrevalCode from '../../../helpers/getPrevalCode';

const theme = createMuiTheme({
  overrides: {
    MuiExpansionPanelDetails: {
      root: {
        background: '#ECECEC',
        padding: '0',
      },
    },
    MuiExpansionPanelSummary: {
      expandIcon: {
        transform: 'translateY(-50%) rotate(0)',
      },

      expandIconExpanded: {
        transform: 'translateY(-50%) rotate(0)',
      },
    },
  },
});

export default class extends Component {
  state = {
    expanded: false,
    innerText: '',
  };

  componentDidMount() {
    const innerText = getPrevalCode(this.props.codeBlock);
    this.setState({ innerText });
  }

  render() {
    const { expanded, innerText } = this.state;
    let CodeIcon = this.props.codeBlock ? (
      <ExpandIcon onClick={() => this.setState({ expanded: !expanded })}>
        <Icon>code</Icon>
      </ExpandIcon>
    ) : (
      ''
    );
    return (
      <ThemeProvider theme={theme}>
        <ExpansionPanel expanded={expanded}>
          <ExpansionPanelSummary
            expandIcon={CodeIcon}
            className="expansionPanelSummary"
          >
            {this.props.title ? (
              <Title title={this.props.title} subtitle={this.props.subtitle} />
            ) : (
              ''
            )}
          </ExpansionPanelSummary>

          <ExpansionPanelDetails>
            {innerText ? (
              <ExpansionPanelDetailsWrapper>
                <CopyToClipboard
                  text={innerText}
                  onCopy={() => notification('success', 'coppied')}
                >
                  <Fab>
                    <Icon>content_copy</Icon>
                  </Fab>
                </CopyToClipboard>
                <CodeMirror
                  value={innerText}
                  options={{
                    lineNumbers: true,
                    theme: 'material',
                    readOnly: true,
                    tabSize: 4,
                    mode: 'javascript',
                  }}
                  preserveScrollPosition
                />
              </ExpansionPanelDetailsWrapper>
            ) : (
              ''
            )}
          </ExpansionPanelDetails>
        </ExpansionPanel>
      </ThemeProvider>
    );
  }
}
