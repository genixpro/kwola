import React from 'react';
import { LayoutContentWrapper } from './layoutWrapper.style';

export default props => (
  <LayoutContentWrapper
    className={
      props.className != null
        ? `${props.className} layoutContentWrapper`
        : 'layoutContentWrapper'
    }
    {...props}
    style={props.style}
  >
    {props.children}
  </LayoutContentWrapper>
);
