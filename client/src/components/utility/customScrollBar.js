import React from 'react';
import Scrollbar from 'react-smooth-scrollbar';
export default ({ id, style, children, className }) => (
  <Scrollbar
    id={id}
    className={className}
    style={style}
    continuousScrolling={true}
  >
    {children}
  </Scrollbar>
);
