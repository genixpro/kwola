import React from 'react';
import PropTypes from 'prop-types';
import { withStyles } from '@material-ui/core/styles';
import Input, { InputLabel } from '../../../components/uielements/input';
import { FormHelperText } from '../../../components/uielements/form';

import { Form, Select, FormControls } from './select.style';

class NativeSelect extends React.Component {
  state = {
    age: '',
    name: 'hai',
  };

  handleChange = name => event => {
    this.setState({ [name]: event.target.value });
  };

  render() {
    const { classes } = this.props;

    return (
      <Form>
        <FormControls>
          <InputLabel htmlFor="age-native-simple">Age</InputLabel>
          <Select
            native
            value={this.state.age}
            onChange={this.handleChange('age')}
            input={<Input id="age-native-simple" />}
          >
            <option value="" />
            <option value={10}>Ten</option>
            <option value={20}>Twenty</option>
            <option value={30}>Thirty</option>
          </Select>
        </FormControls>
        <FormControls>
          <InputLabel htmlFor="age-native-helper">Age</InputLabel>
          <Select
            native
            value={this.state.age}
            onChange={this.handleChange('age')}
            input={<Input id="age-native-helper" />}
          >
            <option value="" />
            <option value={10}>Ten</option>
            <option value={20}>Twenty</option>
            <option value={30}>Thirty</option>
          </Select>
          <FormHelperText>Some important helper text</FormHelperText>
        </FormControls>
        <FormControls>
          <Select
            native
            value={this.state.age}
            onChange={this.handleChange('age')}
            className={classes.selectEmpty}
          >
            <option value="">None</option>
            <option value={10}>Ten</option>
            <option value={20}>Twenty</option>
            <option value={30}>Thirty</option>
          </Select>
          <FormHelperText>Without label</FormHelperText>
        </FormControls>
        <FormControls disabled>
          <InputLabel htmlFor="name-native-disabled">Name</InputLabel>
          <Select
            native
            value={this.state.name}
            onChange={this.handleChange('name')}
            input={<Input id="name-native-disabled" />}
          >
            <option value="" />
            <optgroup label="Author">
              <option value="hai">Hai</option>
            </optgroup>
            <optgroup label="Contributors">
              <option value="olivier">Olivier</option>
              <option value="kevin">Kevin</option>
            </optgroup>
          </Select>
          <FormHelperText>Disabled</FormHelperText>
        </FormControls>
        <FormControls error>
          <InputLabel htmlFor="name-native-error">Name</InputLabel>
          <Select
            native
            value={this.state.name}
            onChange={this.handleChange('name')}
            input={<Input id="name-native-error" />}
          >
            <option value="" />
            <optgroup label="Author">
              <option value="hai">Hai</option>
            </optgroup>
            <optgroup label="Contributors">
              <option value="olivier">Olivier</option>
              <option value="kevin">Kevin</option>
            </optgroup>
          </Select>
          <FormHelperText>Error</FormHelperText>
        </FormControls>
        <FormControls>
          <InputLabel htmlFor="name-input">Name</InputLabel>
          <Input id="name-input" />
          <FormHelperText>Alignment with an input</FormHelperText>
        </FormControls>
        <FormControls>
          <InputLabel htmlFor="uncontrolled-native">Name</InputLabel>
          <Select
            native
            defaultValue={30}
            input={<Input id="uncontrolled-native" />}
          >
            <option value="" />
            <option value={10}>Ten</option>
            <option value={20}>Twenty</option>
            <option value={30}>Thirty</option>
          </Select>
          <FormHelperText>Uncontrolled</FormHelperText>
        </FormControls>
      </Form>
    );
  }
}

NativeSelect.propTypes = {
  classes: PropTypes.object.isRequired,
};

export default withStyles({})(NativeSelect);
