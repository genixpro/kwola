import React from 'react';
import PropTypes from 'prop-types';
import { withStyles } from '@material-ui/core/styles';
import Typography from '../../../components/uielements/typography/index.js';
import Icon from '../../../components/uielements/icon/index.js';
import { MobileStepper, MobileStepperText } from './stepper.style';
import Button from '../../../components/uielements/button';

const styles = theme => ({});

class TextMobileStepper extends React.Component {
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
    const { classes, theme } = this.props;

    return (
      <div className={classes.root}>
        <MobileStepperText square elevation={0}>
          <Typography>Step {this.state.activeStep + 1} of 6</Typography>
        </MobileStepperText>
        <MobileStepper
          variant="text"
          steps={6}
          position="static"
          activeStep={this.state.activeStep}
          className={classes.mobileStepper}
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
      </div>
    );
  }
}

TextMobileStepper.propTypes = {
  classes: PropTypes.object.isRequired,
  theme: PropTypes.object.isRequired,
};

export default withStyles(styles, { withTheme: true })(TextMobileStepper);
