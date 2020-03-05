import React from 'react';

const rtl = document.getElementsByTagName('body')[0].getAttribute('dir');
const withDirection = Component => props => {
  return <Component {...props} data-rtl={rtl} />;
};

export default withDirection;
export { rtl };
