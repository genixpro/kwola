import React from 'react';
import PropTypes from 'prop-types';
import { withStyles } from '@material-ui/core/styles';
import Input, { InputLabel } from '../../../components/uielements/input';
import { Container, FormControl, FormHelperText } from './textfield.style';

class ComposedTextField extends React.Component {
  state = {
    name: 'Composed TextField',
  };

  handleChange = event => {
    this.setState({ name: event.target.value });
  };

  render() {
    return (
      <Container>
        <FormControl>
          <InputLabel htmlFor="name-simple">Name</InputLabel>
          <Input
            id="name-simple"
            value={this.state.name}
            onChange={this.handleChange}
          />
        </FormControl>
        <FormControl>
          <InputLabel htmlFor="name-helper">Name</InputLabel>
          <Input
            id="name-helper"
            value={this.state.name}
            onChange={this.handleChange}
          />
          <FormHelperText>Some important helper text</FormHelperText>
        </FormControl>
        <FormControl disabled>
          <InputLabel htmlFor="name-disabled">Name</InputLabel>
          <Input
            id="name-disabled"
            value={this.state.name}
            onChange={this.handleChange}
          />
          <FormHelperText>Disabled</FormHelperText>
        </FormControl>
        <FormControl error>
          <InputLabel htmlFor="name-error">Name</InputLabel>
          <Input
            id="name-error"
            value={this.state.name}
            onChange={this.handleChange}
          />
          <FormHelperText>Error</FormHelperText>
        </FormControl>
      </Container>
    );
  }
}

ComposedTextField.propTypes = {
  classes: PropTypes.object.isRequired,
};

export default withStyles({})(ComposedTextField);
