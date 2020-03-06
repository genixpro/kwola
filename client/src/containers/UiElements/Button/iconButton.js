import React from 'react';
import IconButton from '../../../components/uielements/iconbutton';
import Icon from '../../../components/uielements/icon/index.js';

const IconButtons = () => (
  <div>
    <IconButton aria-label="Delete">
      <Icon>delete</Icon>
    </IconButton>
    <IconButton aria-label="Delete" disabled color="primary">
      <Icon>delete</Icon>
    </IconButton>
    <IconButton color="secondary" aria-label="Add an alarm">
      <Icon>alarm</Icon>
    </IconButton>
    <IconButton color="inherit" aria-label="Add to shopping cart">
      <Icon>add_shopping_cart</Icon>
    </IconButton>
    <IconButton color="primary" aria-label="Add to shopping cart">
      <Icon>add_shopping_cart</Icon>
    </IconButton>
  </div>
);

export default IconButtons;
