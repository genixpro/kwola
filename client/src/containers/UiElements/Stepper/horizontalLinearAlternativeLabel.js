import React from 'react';
import PropTypes from 'prop-types';
import { withStyles } from '@material-ui/core/styles';
import Typography from '../../../components/uielements/typography';
import Stepper, {
  Step,
  StepLabel,
} from '../../../components/uielements/stepper';
import {
  Root,
  ButtonWrapper,
  StepperContent,
  ButtonContainer,
} from './stepper.style';
import Button from '../../../components/uielements/button';

function getSteps() {
  return [
    'Select master blaster campaign settings',
    'Create an ad group',
    'Create an ad',
  ];
}

function getStepContent(stepIndex) {
  switch (stepIndex) {
    case 0:
      return 'Select campaign settings...';
    case 1:
      return 'What is an ad group anyways?';
    case 2:
      return 'This is the bit I really care about!';

    default:
      return 'Uknown stepIndex';
  }
}

class HorizontalLabelPositionBelowStepper extends React.Component {
  state = {
    activeStep: 0,
  };

  handleNext = () => {
    const { activeStep } = this.state;
    this.setState({
      activeStep: activeStep + 1,
    });
  };

  handleBack = () => {
    const { activeStep } = this.state;
    this.setState({
      activeStep: activeStep - 1,
    });
  };

  handleReset = () => {
    this.setState({
      activeStep: 0,
    });
  };

  render() {
    const steps = getSteps();
    const { activeStep } = this.state;

    return (
      <Root>
        <Stepper activeStep={activeStep} alternativeLabel>
          {steps.map(label => {
            return (
              <Step key={label}>
                <StepLabel>{label}</StepLabel>
              </Step>
            );
          })}
        </Stepper>
        <StepperContent>
          {this.state.activeStep === steps.length ? (
            <div>
              <Typography className="instructions">
                All steps completed - you&quot;re finished
              </Typography>
              <ButtonWrapper>
                <Button onClick={this.handleReset}>Reset</Button>
              </ButtonWrapper>
            </div>
          ) : (
            <div>
              <Typography className="instructions">
                {getStepContent(activeStep)}
              </Typography>
              <ButtonContainer>
                <ButtonWrapper>
                  <Button disabled={activeStep === 0} onClick={this.handleBack}>
                    Back
                  </Button>
                </ButtonWrapper>
                <ButtonWrapper>
                  <Button color="primary" onClick={this.handleNext}>
                    {activeStep === steps.length - 1 ? 'Finish' : 'Next'}
                  </Button>
                </ButtonWrapper>
              </ButtonContainer>
            </div>
          )}
        </StepperContent>
      </Root>
    );
  }
}

HorizontalLabelPositionBelowStepper.propTypes = {
  classes: PropTypes.object,
};

export default withStyles({})(HorizontalLabelPositionBelowStepper);
