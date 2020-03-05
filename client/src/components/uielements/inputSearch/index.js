import React, { Component } from 'react';
import { InputSearch } from './inputSearch.style';
import notification from '../../notification';

export default class extends Component {
  state = {
    value: this.props.defaultValue || ''
  };

  onKeyPress = event => {
    if (event.key === 'Enter') {
      event.preventDefault();
      const { value } = this.state;
      if (value && value.length > 0 && this.props.onSearch) {
        this.props.onSearch(value);
        if (this.props.clearOnSearch) {
          this.setState({ value: '' });
        }
      } else {
        notification('error', 'Please type something');
      }
    }
  };
  onChange = event => {
    event.preventDefault();
    const value = event.target.value;
    this.setState({ value });
    if (this.props.onChange) {
      this.props.onChange(value);
    }
  };
  render() {
    const { alwaysDefaultValue, defaultValue, className } = this.props;
    const value = alwaysDefaultValue ? defaultValue : this.state.value;
    return (
      <InputSearch
        className={className}
        onKeyPress={this.onKeyPress}
        onChange={this.onChange}
        value={value}
        label={this.props.label}
        placeholder={this.props.placeholder}
        fullWidth={this.props.fullWidth}
        id={this.props.id}
        disableUnderline={this.props.disableUnderline}
        startAdornment={this.props.startAdornment}
      />
    );
  }
}
