import React, { Component } from 'react';
import { findDOMNode } from 'react-dom';
import Button from '../uielements/button';
import Popover from '../uielements/popover';
import { GithubPicker } from '../uielements/reactColor';
import ColorChooserDropdown from './colorChooser.style';

export default class ColorChoser extends Component {
  constructor(props) {
    super(props);
    this.handleVisibleChange = this.handleVisibleChange.bind(this);
    this.hide = this.hide.bind(this);
    this.state = {
      visible: false,
      anchorEl: null,
    };
  }
  hide() {
    this.setState({ visible: false });
  }
  handleVisibleChange() {
    this.setState({
      visible: !this.state.visible,
      anchorEl: findDOMNode(this.button),
    });
  }
  render() {
    const { colors, seectedColor, changeColor } = this.props;
    const colorsIndex = {};
    colors.forEach((color, index) => {
      colorsIndex[color.toLowerCase()] = index;
    });
    const content = () => (
      <ColorChooserDropdown>
        <GithubPicker
          colors={colors}
          color={colors[seectedColor]}
          className="mateColorWrapper"
          triangle="hide"
          disableAlpha={false}
          onChange={col => {
            changeColor(colorsIndex[col.hex]);
          }}
        />
      </ColorChooserDropdown>
    );
    return (
      <div>
        <Button
          variant="contained"
          className="ColorChooser"
          style={{ backgroundColor: colors[seectedColor] }}
          ref={node => {
            this.button = node;
          }}
          onClick={this.handleVisibleChange}
        >
          {' '}
        </Button>
        <Popover
          open={this.state.visible}
          anchorEl={this.state.anchorEl}
          // anchorReference={'anchorEl'}
          anchorPosition={{ top: 200, left: 400 }}
          onClose={this.hide}
          anchorOrigin={{ vertical: 'top', horizontal: 'center' }}
          transformOrigin={{ vertical: 'bottom', horizontal: 'center' }}
        >
          {content()}
        </Popover>
      </div>
    );
  }
}
