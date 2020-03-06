import React from 'react';
import { Button, Input } from './button.style';

const RaisedButtons = () => (
  <div>
    <Button variant="contained">Default</Button>
    <Button variant="contained" color="primary">
      Primary
    </Button>
    <Button variant="contained" color="secondary">
      Secondary
    </Button>
    <Button variant="contained" color="inherit">
      Inherit
    </Button>
    <Button variant="contained" color="secondary" disabled>
      Disabled
    </Button>
    <Input accept="jpg,jpeg,JPG,JPEG" id="file" multiple type="file" />
    <label htmlFor="file">
      <Button variant="contained" component="span">
        Upload
      </Button>
    </label>
  </div>
);

export default RaisedButtons;
