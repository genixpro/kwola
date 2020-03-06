import React from 'react';
import Icon from '../../../components/uielements/icon/index.js';
import { Button } from './button.style';

const FloatingActionButtons = () => (
  <div>
    <Button color="primary" aria-label="add">
      <Icon>add</Icon>
    </Button>
    <Button color="secondary" aria-label="edit">
      <Icon>mode_edit</Icon>
    </Button>
  </div>
);

export default FloatingActionButtons;
