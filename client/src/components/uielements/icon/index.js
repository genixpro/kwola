import React, { Component } from 'react';
import Icon from '@material-ui/core/Icon';

class FontIcon extends Component {
  render() {
    const { className } = this.props;
    return <i className={`zmdi zmdi-${className}`} />;
  }
}

export { FontIcon };
export default Icon;
