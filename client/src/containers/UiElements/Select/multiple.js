import React from 'react';
import PropTypes from 'prop-types';
import { withStyles } from '@material-ui/core/styles';
import Input, { InputLabel } from '../../../components/uielements/input';
import { MenuItem } from '../../../components/uielements/menus';
import { Form, Select, FormControls } from './select.style';

const ITEM_HEIGHT = 48;
const ITEM_PADDING_TOP = 8;

const names = [
  'Oliver Hansen',
  'Van Henry',
  'April Tucker',
  'Ralph Hubbard',
  'Omar Alexander',
  'Carlos Abbott',
  'Miriam Wagner',
  'Bradley Wilkerson',
  'Virginia Andrews',
  'Kelly Snyder',
];

class MultipleSelect extends React.Component {
  state = {
    name: [],
  };

  handleChange = event => {
    this.setState({ name: event.target.value });
  };

  render() {
    return (
      <Form>
        <FormControls style={{ width: '100%' }}>
          <InputLabel htmlFor="name-multiple">Name</InputLabel>
          <Select
            multiple
            value={this.state.name}
            onChange={this.handleChange}
            input={<Input id="name-multiple" />}
            MenuProps={{
              PaperProps: {
                style: {
                  maxHeight: ITEM_HEIGHT * 4.5 + ITEM_PADDING_TOP,
                  width: 200,
                },
              },
            }}
          >
            {names.map(name => (
              <MenuItem
                key={name}
                value={name}
                style={{
                  fontWeight:
                    this.state.name.indexOf(name) !== -1 ? '500' : '400',
                }}
              >
                {name}
              </MenuItem>
            ))}
          </Select>
        </FormControls>
      </Form>
    );
  }
}

MultipleSelect.propTypes = {
  classes: PropTypes.object.isRequired,
};

export default withStyles({})(MultipleSelect);
