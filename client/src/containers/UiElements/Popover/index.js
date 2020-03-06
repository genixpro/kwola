import React, { Component } from 'react';
import { findDOMNode } from 'react-dom';
import PropTypes from 'prop-types';
import Typography from '../../../components/uielements/typography';
import StyledPopover, { StyledButton } from './popover.style';

class PopoverExamples extends Component {
  state = {
    open: false,
    anchorEl: null,
    anchorOriginVertical: 'bottom',
    anchorOriginHorizontal: 'center',
    transformOriginVertical: 'top',
    transformOriginHorizontal: 'center',
    positionTop: 200, // Just so the popover can be spotted more easily
    positionLeft: 400, // Same as above
    anchorReference: 'anchorEl',
  };

  handleChange = key => (event, value) => {
    this.setState({
      [key]: value,
    });
  };

  handleNumberInputChange = key => event => {
    this.setState({
      [key]: parseInt(event.target.value, 10),
    });
  };

  handleClickButton = () => {
    this.setState({
      open: true,
      anchorEl: findDOMNode(this.button),
    });
  };

  handleRequestClose = () => {
    this.setState({
      open: false,
    });
  };

  button = null;

  render() {
    // const { classes } = this.props;
    const {
      open,
      anchorEl,
      anchorOriginVertical,
      anchorOriginHorizontal,
      transformOriginVertical,
      transformOriginHorizontal,
      positionTop,
      positionLeft,
      anchorReference,
    } = this.state;

    return (
      <div>
        <StyledButton
          ref={node => {
            this.button = node;
          }}
          className="hello"
          variant="contained"
          // className={classes.button}
          onClick={this.handleClickButton}
        >
          Open Popover
        </StyledButton>
        <StyledPopover
          open={open}
          anchorEl={anchorEl}
          anchorReference={anchorReference}
          anchorPosition={{ top: positionTop, left: positionLeft }}
          onClose={this.handleRequestClose}
          className="styledSupport"
          anchorOrigin={{
            vertical: anchorOriginVertical,
            horizontal: anchorOriginHorizontal,
          }}
          transformOrigin={{
            vertical: transformOriginVertical,
            horizontal: transformOriginHorizontal,
          }}
        >
          <Typography>The content of the Popover.</Typography>
        </StyledPopover>
      </div>
    );
  }
}

PopoverExamples.propTypes = {
  classes: PropTypes.object.isRequired,
};

export default PopoverExamples;
