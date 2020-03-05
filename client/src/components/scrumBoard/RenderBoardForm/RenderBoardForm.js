import React from 'react';
import TextField from '@material-ui/core/TextField';
import MenuItem from '@material-ui/core/MenuItem';
import InputLabel from '@material-ui/core/InputLabel';
import Select from '@material-ui/core/Select';
import Switch from '@material-ui/core/Switch';
import Tooltip from '@material-ui/core/Tooltip';
import Button from '@material-ui/core/Button';
import FormGroup from '@material-ui/core/FormGroup';
import FormControl from '@material-ui/core/FormControl';
import FormControlLabel from '@material-ui/core/FormControlLabel';
import ThumbIcon from './08-icon.svg';
import { Wrapper } from './RenderBoardForm.style';

const categories = [
  {
    value: 'software',
    label: 'Software',
  },
  {
    value: 'service-desk',
    label: 'Service Desk',
  },
  {
    value: 'OPs',
    label: 'OPs',
  },
  {
    value: 'Business',
    label: 'Business',
  },
  {
    value: 'General',
    label: 'General',
  },
];

export default function BoardForm({
  handleSubmit,
  values,
  handleChange,
  handleBlur,
  errors,
  touched,
}) {
  const { title, category, open_to_company, open_to_members } = values;

  return (
    <Wrapper>
      <form onSubmit={handleSubmit}>
        <FormGroup>
          <TextField
            label="Project Name*"
            name="title"
            value={title}
            onChange={handleChange}
            onBlur={handleBlur}
            helperText={errors.title && touched.title && errors.title}
            margin="normal"
          />
        </FormGroup>

        <FormGroup>
          <FormControl>
            <InputLabel htmlFor="category">Project Category*</InputLabel>
            <Select
              value={category}
              onChange={handleChange}
              inputProps={{
                name: 'category',
                id: 'category',
              }}
            >
              {categories.map(option => (
                <MenuItem key={option.value} value={option.value}>
                  {option.label}
                </MenuItem>
              ))}
            </Select>
          </FormControl>
        </FormGroup>

        <FormGroup style={{ marginTop: '30px' }}>
          <FormControlLabel
            control={
              <Switch
                checked={open_to_company}
                onChange={handleChange}
                value={open_to_company}
                name="open_to_company"
                color="primary"
              />
            }
            label="Public To RedQ Studio"
          />
          <FormControlLabel
            control={
              <Switch
                checked={open_to_members}
                onChange={handleChange}
                value={open_to_members}
                name="open_to_members"
                color="primary"
              />
            }
            label="Private To Project Members"
          />
        </FormGroup>

        <div className="field-container">
          <img
            src={ThumbIcon}
            alt="Project"
            width={40}
            height={40}
            style={{ marginRight: 10, borderRadius: 6 }}
          />
          <Tooltip title="Please Implements Your Own Avatar Methods">
            <div>Select Avatar</div>
          </Tooltip>
        </div>

        <Button variant="contained" color="primary" type="submit" size="large">
          {!values.editing ? 'Create Project' : 'Update Task'}
        </Button>
      </form>
    </Wrapper>
  );
}
