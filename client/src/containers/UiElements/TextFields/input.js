import React from 'react';
import PropTypes from 'prop-types';
import { withStyles } from '@material-ui/core/styles';

import { Container, Input } from './textfield.style';

const styles = theme => ({});

function Inputs(props) {
  return (
    <Container>
      <Input
        defaultValue="Hello world"
        inputProps={{
          'aria-label': 'Description',
        }}
      />
      <Input
        placeholder="Placeholder"
        inputProps={{
          'aria-label': 'Description',
        }}
      />
      <Input
        value="Disabled"
        disabled
        inputProps={{
          'aria-label': 'Description',
        }}
      />
      <Input
        defaultValue="Error"
        error
        inputProps={{
          'aria-label': 'Description',
        }}
      />
    </Container>
  );
}

Inputs.propTypes = {
  classes: PropTypes.object.isRequired,
};

export default withStyles(styles)(Inputs);
