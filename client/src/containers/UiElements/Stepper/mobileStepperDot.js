import React from 'react';
import PropTypes from 'prop-types';
import { withStyles } from '@material-ui/core/styles';
import Icon from '../../../components/uielements/icon/index.js';
import Button from '../../../components/uielements/button';
import { MobileStepper } from './stepper.style';

const styles = theme => ({});

class DotsMobileStepper extends React.Component {
  state = {
    activeStep: 0,
  };

  handleNext = () => {
    this.setState({
      activeStep: this.state.activeStep + 1,
    });
  };

  handleBack = () => {
    this.setState({
      activeStep: this.state.activeStep - 1,
    });
  };

  render() {
    const { theme } = this.props;

    return (
      <MobileStepper
        variant="dots"
        steps={6}
        position="static"
        activeStep={this.state.activeStep}
        nextButton={
          <Button
            size="small"
            onClick={this.handleNext}
            disabled={this.state.activeStep === 5}
          >
            Next
            {theme.direction === 'rtl' ? (
              <Icon>keyboard_arrow_left</Icon>
            ) : (
              <Icon>keyboard_arrow_right</Icon>
            )}
          </Button>
        }
        backButton={
          <Button
            size="small"
            onClick={this.handleBack}
            disabled={this.state.activeStep === 0}
          >
            {theme.direction === 'rtl' ? (
              <Icon>keyboard_arrow_right</Icon>
            ) : (
              <Icon>keyboard_arrow_left</Icon>
            )}
            Back
          </Button>
        }
      />
    );
  }
}

DotsMobileStepper.propTypes = {
  classes: PropTypes.object.isRequired,
  theme: PropTypes.object.isRequired,
};

export default withStyles(styles, { withTheme: true })(DotsMobileStepper);
