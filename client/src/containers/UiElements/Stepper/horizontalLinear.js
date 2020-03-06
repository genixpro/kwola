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
  StepperContent,
  ButtonWrapper,
  ButtonContainer,
} from './stepper.style';
import Button from '../../../components/uielements/button';

function getSteps() {
  return ['Select campaign settings', 'Create an ad group', 'Create an ad'];
}

function getStepContent(step) {
  switch (step) {
    case 0:
      return 'Select campaign settings...';
    case 1:
      return 'What is an ad group anyways?';
    case 2:
      return 'This is the bit I really care about!';
    default:
      return 'Unknown step';
  }
}

class HorizontalLinearStepper extends React.Component {
  static propTypes = {
    classes: PropTypes.object,
  };

  state = {
    activeStep: 0,
    skipped: new Set(),
  };

  isStepOptional = step => {
    return step === 1;
  };

  isStepSkipped(step) {
    return this.state.skipped.has(step);
  }

  handleNext = () => {
    const { activeStep } = this.state;
    let { skipped } = this.state;
    if (this.isStepSkipped(activeStep)) {
      skipped = new Set(skipped.values());
      skipped.delete(activeStep);
    }
    this.setState({
      activeStep: activeStep + 1,
      skipped,
    });
  };

  handleBack = () => {
    const { activeStep } = this.state;
    this.setState({
      activeStep: activeStep - 1,
    });
  };

  handleSkip = () => {
    const { activeStep } = this.state;
    if (!this.isStepOptional(activeStep)) {
      // You probably want to guard against something like this,
      // it should never occur unless someone's actively trying to break something.
      throw new Error("You can't skip a step that isn't optional.");
    }
    const skipped = new Set(this.state.skipped.values());
    skipped.add(activeStep);
    this.setState({
      activeStep: this.state.activeStep + 1,
      skipped,
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
        <Stepper activeStep={activeStep}>
          {steps.map((label, index) => {
            const props = {};
            if (this.isStepOptional(index)) {
              props.optional = <Typography type="caption">Optional</Typography>;
            }
            if (this.isStepSkipped(index)) {
              props.completed = false;
            }
            return (
              <Step key={label} {...props}>
                <StepLabel>{label}</StepLabel>
              </Step>
            );
          })}
        </Stepper>
        <StepperContent>
          {activeStep === steps.length ? (
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
                {this.isStepOptional(activeStep) && (
                  <ButtonWrapper>
                    <Button color="primary" onClick={this.handleSkip}>
                      Skip
                    </Button>
                  </ButtonWrapper>
                )}
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

export default withStyles({})(HorizontalLinearStepper);
