import React, { Component } from 'react';
import { EmptyComponent } from './emptyComponent.style';

export default class extends Component {
  render() {
    const value = this.props.value || 'Please include Config';
    return (
      <EmptyComponent>
        <span>{value}</span>
      </EmptyComponent>
    );
  }
}
