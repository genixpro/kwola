import React from 'react';
import PropTypes from 'prop-types';
import { withStyles } from '@material-ui/core/styles';
import { MenuItem } from '../../../components/uielements/menus';
import { Form, TextField } from './textfield.style';

const currencies = [
  {
    value: 'argentina',
    label: 'Argentina',
  },
  {
    value: 'australia',
    label: 'Australia',
  },
  {
    value: 'brazil',
    label: 'Brazil',
  },
  {
    value: 'france',
    label: 'France',
  },
];

class TextFields extends React.Component {
  state = {
    name: 'Cat in the Hat',
    age: '',
    multiline: 'Controlled',
    currency: 'EUR',
  };

  handleChange = name => event => {
    this.setState({
      [name]: event.target.value,
    });
  };

  render() {
    return (
      <Form noValidate autoComplete="off">
        <TextField
          id="name"
          label="Name"
          value={this.state.name}
          onChange={this.handleChange('name')}
          margin="normal"
        />
        <TextField
          id="uncontrolled"
          label="Uncontrolled"
          defaultValue="foo"
          margin="normal"
        />
        <TextField
          required
          id="required"
          label="Required"
          defaultValue="Hello World"
          margin="normal"
        />
        <TextField
          error
          id="error"
          label="Error"
          defaultValue="Hello World"
          margin="normal"
        />
        <TextField
          id="password1"
          label="Password"
          type="password"
          autoComplete="current-password"
          margin="normal"
        />
        <TextField
          id="multiline-flexible"
          label="Multiline"
          multiline
          rowsMax="4"
          value={this.state.multiline}
          onChange={this.handleChange('multiline')}
          margin="normal"
        />
        <TextField
          id="multiline-static"
          label="Multiline"
          multiline
          rows="4"
          defaultValue="Default Value"
          margin="normal"
        />
        <TextField
          id="helperText"
          label="Helper text"
          defaultValue="Default Value"
          helperText="Some important text"
          margin="normal"
        />
        <TextField
          label="With placeholder"
          placeholder="Placeholder"
          margin="normal"
        />
        <TextField
          label="With placeholder multiline"
          placeholder="Placeholder"
          multiline
          margin="normal"
        />
        <TextField
          id="number"
          label="Number"
          value={this.state.age}
          onChange={this.handleChange('age')}
          type="number"
          InputLabelProps={{
            shrink: true,
          }}
          margin="normal"
        />
        <TextField
          id="search"
          label="Search field"
          type="search"
          margin="normal"
        />
        <TextField
          id="select-currency"
          select
          label="Select"
          value={this.state.currency}
          onChange={this.handleChange('currency')}
          SelectProps={{
            MenuProps: {
              className: 'menu',
            },
          }}
          helperText="Please select your currency"
          margin="normal"
        >
          {currencies.map(option => (
            <MenuItem key={option.value} value={option.value}>
              {option.label}
            </MenuItem>
          ))}
        </TextField>
        <TextField
          id="select-currency-native"
          select
          label="Native select"
          value={this.state.currency}
          onChange={this.handleChange('currency')}
          SelectProps={{
            native: true,
            MenuProps: {
              className: 'menu',
            },
          }}
          margin="normal"
        >
          {currencies.map(option => (
            <option key={option.value} value={option.value}>
              {option.label}
            </option>
          ))}
        </TextField>
        <TextField
          id="full-width"
          label="Label"
          InputLabelProps={{
            shrink: true,
          }}
          placeholder="Placeholder"
          helperText="Full width!"
          margin="normal"
          style={{ margin: 8 }}
          fullWidth
        />
      </Form>
    );
  }
}

TextFields.propTypes = {
  classes: PropTypes.object.isRequired,
};

export default withStyles({})(TextFields);
