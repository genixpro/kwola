import React from 'react';
import PropTypes from 'prop-types';
import { withStyles } from '@material-ui/core/styles';
import Typography from '../../../components/uielements/typography';
import Stepper, { Step } from '../../../components/uielements/stepper';
import {
  Root,
  ButtonWrapper,
  StepperContent,
  ButtonContainer,
} from './stepper.style';
import Button from '../../../components/uielements/button';

function getSteps() {
  return ['Select campaign settings', 'Create an ad group', 'Create an ad'];
}

function getStepContent(step) {
  switch (step) {
    case 0:
      return 'Step 1: Select campaign settings...';
    case 1:
      return 'Step 2: What is an ad group anyways?';
    case 2:
      return 'Step 3: This is the bit I really care about!';
    default:
      return 'Unknown step';
  }
}

class HorizontalNonLinearStepper extends React.Component {
  state = {
    activeStep: 0,
    completed: {},
  };

  completedSteps() {
    return Object.keys(this.state.completed).length;
  }

  totalSteps = () => {
    return getSteps().length;
  };

  isLastStep() {
    return this.state.activeStep === this.totalSteps() - 1;
  }

  allStepsCompleted() {
    return this.completedSteps() === this.totalSteps();
  }

  handleNext = () => {
    let activeStep;

    if (this.isLastStep() && !this.allStepsCompleted()) {
      const steps = getSteps();
      activeStep = steps.findIndex((step, i) => !(i in this.state.completed));
    } else {
      activeStep = this.state.activeStep + 1;
    }
    this.setState({
      activeStep,
    });
  };

  handleBack = () => {
    const { activeStep } = this.state;
    this.setState({
      activeStep: activeStep - 1,
    });
  };

  handleStep = step => () => {
    this.setState({
      activeStep: step,
    });
  };

  handleComplete = () => {
    const { completed } = this.state;
    completed[this.state.activeStep] = true;
    this.setState({
      completed,
    });
    this.handleNext();
  };

  handleReset = () => {
    this.setState({
      activeStep: 0,
      completed: {},
    });
  };

  render() {
    const steps = getSteps();
    const { activeStep } = this.state;

    return (
      <Root>
        <Stepper nonLinear activeStep={activeStep}>
          {steps.map((label, index) => (
            <Step key={label}>
              <ButtonWrapper>
                <Button
                  onClick={this.handleStep(index)}
                  completed={this.state.completed[index]}
                >
                  {label}
                </Button>
              </ButtonWrapper>
            </Step>
          ))}
        </Stepper>
        <StepperContent>
          {this.allStepsCompleted() ? (
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
                    Next
                  </Button>
                </ButtonWrapper>
                {activeStep !== steps.length &&
                  (this.state.completed[this.state.activeStep] ? (
                    <Typography type="caption" className="completed">
                      Step {activeStep + 1} already completed
                    </Typography>
                  ) : (
                    <ButtonWrapper>
                      <Button color="primary" onClick={this.handleComplete}>
                        {this.completedSteps() === this.totalSteps() - 1
                          ? 'Finish'
                          : 'Complete Step'}
                      </Button>
                    </ButtonWrapper>
                  ))}
              </ButtonContainer>
            </div>
          )}
        </StepperContent>
      </Root>
    );
  }
}

HorizontalNonLinearStepper.propTypes = {
  classes: PropTypes.object,
};

export default withStyles({})(HorizontalNonLinearStepper);
