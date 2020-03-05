import React from 'react';
import PropTypes from 'prop-types';
import { MenuItem } from '../uielements/menus';
import { FormControlLabel } from '../uielements/form';
import Checkbox from '../../components/uielements/checkbox';
import { Form, TextField } from './billingForm.style';

const countries = [
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

class BillingForm extends React.Component {
  state = {
    name: '',
    age: '',
    multiline: 'Controlled',
    country: 'argentina',
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
          required
          id="firstName"
          className="firstNameField billingFormField"
          label="First Name"
          margin="normal"
        />
        <TextField
          required
          id="lastName"
          className="lastNameField billingFormField"
          label="Last Name"
          margin="normal"
        />
        <TextField
          id="companyName"
          className="companyNameField fullColumnWidth billingFormField"
          label="Company Name"
          margin="normal"
        />
        <TextField
          required
          id="email"
          className="emailField billingFormField"
          label="Email Address"
          type="email"
          margin="normal"
        />
        <TextField
          id="mobileNo"
          className="mobileNoField billingFormField"
          label="Mobile No"
          type="text"
          margin="normal"
        />
        <TextField
          id="select-country"
          className="selectField billingFormField"
          select
          label="Select"
          value={this.state.country}
          onChange={this.handleChange('country')}
          SelectProps={{
            MenuProps: {
              className: 'menu',
            },
          }}
          margin="normal"
        >
          {countries.map(option => (
            <MenuItem key={option.value} value={option.value}>
              {option.label}
            </MenuItem>
          ))}
        </TextField>
        <TextField
          id="city"
          className="cityField billingFormField"
          label="City"
          margin="normal"
        />
        <TextField
          id="address"
          className="addressField fullColumnWidth billingFormField"
          label="Address"
          margin="normal"
        />
        <TextField
          id="apartment"
          className="apartmentField fullColumnWidth billingFormField"
          label="Apartment, suite, unit etc. (optional)"
          margin="normal"
        />

        <FormControlLabel
          className="checkboxField billingFormField"
          control={<Checkbox color="primary" value="checkedA" />}
          label="Create an account?"
        />
      </Form>
    );
  }
}

BillingForm.propTypes = {
  classes: PropTypes.object.isRequired,
};

export default BillingForm;
