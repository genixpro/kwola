import React from 'react';
import { Button } from './button.style';

const DefaultButtons = () => (
  <div>
    <Button>Default</Button>
    <Button color="primary">Primary</Button>
    <Button color="secondary">Secondary</Button>
    <Button color="inherit">Inherit</Button>
    <Button disabled>Disabled</Button>
    <Button href="#flat-buttons">Link</Button>
    <Button disabled href="/">
      Link disabled
    </Button>
    <Button data-something="here I am">Does something</Button>
  </div>
);

export default DefaultButtons;
