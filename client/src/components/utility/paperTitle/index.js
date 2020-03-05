import React from 'react';
import { PageTitle, PageSubTitle } from './paperTitle.style';

export default props => {
  return (
    <PageTitle
      style={props.style}
      className={`${props[`data-single`] ? 'single' : ''} ${props.className}`}
    >
      {props.title ? <h3>{props.title}</h3> : ''}
      {props.subtitle ? <PageSubTitle> {props.subtitle} </PageSubTitle> : ''}
    </PageTitle>
  );
};
