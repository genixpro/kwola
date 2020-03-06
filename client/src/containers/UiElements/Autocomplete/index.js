import React, { Component } from 'react';
import { withStyles } from '@material-ui/core/styles';
import LayoutWrapper from '../../../components/utility/layoutWrapper';
import BasicAutoComplete from './basicAutoComplete';
import Papersheet from '../../../components/utility/papersheet';
import { Row, FullColumn } from '../../../components/utility/rowColumn';

const styles = theme => ({
  container: {
    flexGrow: 1,
    position: 'relative',
    height: 50,
  },
  suggestionsContainerOpen: {
    position: 'absolute',
    marginTop: theme.spacing(1),
    marginBottom: theme.spacing(3),
    left: 0,
    right: 0,
  },
  suggestion: {
    display: 'block',
  },
  suggestionsList: {
    margin: 0,
    padding: 0,
    listStyleType: 'none',
  },
  textField: {
    width: '100%',
  },
});

class IntegrationAutosuggest extends Component {
  render() {
    return (
      <LayoutWrapper>
        <Row>
          <FullColumn>
            <Papersheet
              title="Basic AutoComplete"
              codeBlock="UiElements/Autocomplete/basicAutoComplete.js"
            >
              <BasicAutoComplete {...this.props} />
            </Papersheet>
          </FullColumn>
        </Row>
      </LayoutWrapper>
    );
  }
}

export default withStyles(styles)(IntegrationAutosuggest);
