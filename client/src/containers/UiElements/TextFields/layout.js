import React from 'react';
import PropTypes from 'prop-types';
import { withStyles } from '@material-ui/core/styles';

import { Container, TextField } from './textfield.style';

const TextFieldMargins = props => {
  return (
    <Container>
      <TextField
        label="None"
        id="margin-none"
        defaultValue="Default Value"
        helperText="Some important text"
      />
      <TextField
        label="Dense"
        id="margin-dense"
        defaultValue="Default Value"
        helperText="Some important text"
        margin="dense"
      />
      <TextField
        label="Normal"
        id="margin-normal"
        defaultValue="Default Value"
        helperText="Some important text"
        margin="normal"
      />
    </Container>
  );
};

TextFieldMargins.propTypes = {
  classes: PropTypes.object.isRequired,
};

export default withStyles({})(TextFieldMargins);
