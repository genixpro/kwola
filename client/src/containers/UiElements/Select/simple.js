import React from 'react';
import PropTypes from 'prop-types';
import { withStyles } from '@material-ui/core/styles';
import Input, { InputLabel } from '../../../components/uielements/input';
import { FormHelperText } from '../../../components/uielements/form';

import { MenuItem } from '../../../components/uielements/menus';
import { Form, Select, FormControls } from './select.style';

class SimpleSelect extends React.Component {
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
      <Form autoComplete="off">
        <FormControls>
          <InputLabel htmlFor="age-simple">Age</InputLabel>
          <Select
            value={this.state.age}
            onChange={this.handleChange('age')}
            input={<Input id="age-simple" />}
          >
            <MenuItem value="">
              <em>None</em>
            </MenuItem>
            <MenuItem value={10}>Ten</MenuItem>
            <MenuItem value={20}>Twenty</MenuItem>
            <MenuItem value={30}>Thirty</MenuItem>
          </Select>
        </FormControls>
        <FormControls>
          <InputLabel htmlFor="age-helper">Age</InputLabel>
          <Select
            value={this.state.age}
            onChange={this.handleChange('age')}
            input={<Input id="age-helper" />}
          >
            <MenuItem value="">
              <em>None</em>
            </MenuItem>
            <MenuItem value={10}>Ten</MenuItem>
            <MenuItem value={20}>Twenty</MenuItem>
            <MenuItem value={30}>Thirty</MenuItem>
          </Select>
          <FormHelperText>Some important helper text</FormHelperText>
        </FormControls>
        <FormControls>
          <Select
            value={this.state.age}
            onChange={this.handleChange('age')}
            displayEmpty
            className={classes.selectEmpty}
          >
            <MenuItem value="">
              <em>None</em>
            </MenuItem>
            <MenuItem value={10}>Ten</MenuItem>
            <MenuItem value={20}>Twenty</MenuItem>
            <MenuItem value={30}>Thirty</MenuItem>
          </Select>
          <FormHelperText>Without label</FormHelperText>
        </FormControls>
        <FormControls disabled>
          <InputLabel htmlFor="name-disabled">Name</InputLabel>
          <Select
            value={this.state.name}
            onChange={this.handleChange('name')}
            input={<Input id="name-disabled" />}
          >
            <MenuItem value="">
              <em>None</em>
            </MenuItem>
            <MenuItem value="hai">Hai</MenuItem>
            <MenuItem value="olivier">Olivier</MenuItem>
            <MenuItem value="kevin">Kevin</MenuItem>
          </Select>
          <FormHelperText>Disabled</FormHelperText>
        </FormControls>
        <FormControls error>
          <InputLabel htmlFor="name-error">Name</InputLabel>
          <Select
            value={this.state.name}
            onChange={this.handleChange('name')}
            renderValue={value => `⚠️  - ${value}`}
            input={<Input id="name-error" />}
          >
            <MenuItem value="">
              <em>None</em>
            </MenuItem>
            <MenuItem value="hai">Hai</MenuItem>
            <MenuItem value="olivier">Olivier</MenuItem>
            <MenuItem value="kevin">Kevin</MenuItem>
          </Select>
          <FormHelperText>Error</FormHelperText>
        </FormControls>
        <FormControls>
          <InputLabel htmlFor="name-input">Name</InputLabel>
          <Input id="name-input" />
          <FormHelperText>Alignment with an input</FormHelperText>
        </FormControls>
        <FormControls>
          <InputLabel htmlFor="name-readonly">Name</InputLabel>
          <Select
            value={this.state.name}
            onChange={this.handleChange('name')}
            input={<Input id="name-readonly" readOnly />}
          >
            <MenuItem value="">
              <em>None</em>
            </MenuItem>
            <MenuItem value="hai">Hai</MenuItem>
            <MenuItem value="olivier">Olivier</MenuItem>
            <MenuItem value="kevin">Kevin</MenuItem>
          </Select>
          <FormHelperText>Read only</FormHelperText>
        </FormControls>
        <FormControls>
          <InputLabel htmlFor="age-simple">Age</InputLabel>
          <Select
            value={this.state.age}
            onChange={this.handleChange('age')}
            input={<Input id="age-simple" />}
            autoWidth
          >
            <MenuItem value="">
              <em>None</em>
            </MenuItem>
            <MenuItem value={10}>Ten</MenuItem>
            <MenuItem value={20}>Twenty</MenuItem>
            <MenuItem value={30}>Thirty</MenuItem>
          </Select>
          <FormHelperText>Auto width</FormHelperText>
        </FormControls>
      </Form>
    );
  }
}

SimpleSelect.propTypes = {
  classes: PropTypes.object.isRequired,
};

export default withStyles({})(SimpleSelect);
