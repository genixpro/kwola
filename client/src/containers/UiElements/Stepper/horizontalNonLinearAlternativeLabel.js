import React from 'react';
import PropTypes from 'prop-types';
import { withStyles } from '@material-ui/core/styles';
import Typography from '../../../components/uielements/typography';
import Stepper, {
  Step,
  StepButton,
} from '../../../components/uielements/stepper';
import {
  Root,
  ButtonWrapper,
  StepperContent,
  ButtonContainer,
} from './stepper.style';
import Button from '../../../components/uielements/button';

const styles = theme => ({
  root: {
    width: '90%',
  },
  button: {
    marginRight: theme.spacing(1),
  },
  backButton: {
    marginRight: theme.spacing(1),
  },
  completed: {
    display: 'inline-block',
  },
  instructions: {
    marginTop: theme.spacing(1),
    marginBottom: theme.spacing(1),
  },
});

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

class HorizontalNonLinearAlternativeLabelStepper extends React.Component {
  state = {
    activeStep: 0,
    completed: new Set(),
    skipped: new Set(),
  };

  totalSteps = () => {
    return getSteps().length;
  };

  isStepComplete(step) {
    return this.state.completed.has(step);
  }

  completedSteps() {
    return this.state.completed.size;
  }

  allStepsCompleted() {
    return this.completedSteps() === this.totalSteps() - this.skippedSteps();
  }

  isLastStep() {
    return this.state.activeStep === this.totalSteps() - 1;
  }

  isStepOptional = step => {
    return step === 1;
  };

  isStepSkipped(step) {
    return this.state.skipped.has(step);
  }

  handleSkip = () => {
    const { activeStep } = this.state;
    if (!this.isStepOptional(activeStep)) {
      // You probably want to guard against something like this
      // it should never occur unless someone's actively trying to break something.
      throw new Error("You can't skip a step that isn't optional.");
    }
    const skipped = new Set(this.state.skipped);
    skipped.add(activeStep);
    this.setState({
      activeStep: this.state.activeStep + 1,
      skipped,
    });
  };

  skippedSteps() {
    return this.state.skipped.size;
  }

  handleNext = () => {
    let activeStep;

    if (this.isLastStep() && !this.allStepsCompleted()) {
      // It's the last step, but not all steps have been completed
      // find the first step that has been completed
      const steps = getSteps();
      activeStep = steps.findIndex((step, i) => !this.state.completed.has(i));
    } else {
      activeStep = this.state.activeStep + 1;
    }
    this.setState({
      activeStep,
    });
  };

  handleBack = () => {
    this.setState({
      activeStep: this.state.activeStep - 1,
    });
  };

  handleStep = step => () => {
    this.setState({
      activeStep: step,
    });
  };

  handleComplete = () => {
    const completed = new Set(this.state.completed);
    completed.add(this.state.activeStep);
    this.setState({
      completed,
    });
    /**
     * Sigh... it would be much nicer to replace the following if conditional with
     * `if (!this.allStepsComplete())` however state is not set when we do this,
     * thus we have to resort to not being very DRY.
     */
    if (completed.size !== this.totalSteps() - this.skippedSteps()) {
      this.handleNext();
    }
  };

  handleReset = () => {
    this.setState({
      activeStep: 0,
      completed: new Set(),
      skipped: new Set(),
    });
  };

  render() {
    const steps = getSteps();
    const { activeStep } = this.state;

    return (
      <Root>
        <Stepper alternativeLabel nonLinear activeStep={activeStep}>
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
                <StepButton
                  onClick={this.handleStep(index)}
                  completed={this.isStepComplete(index)}
                >
                  {label}
                </StepButton>
              </Step>
            );
          })}
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
                {this.isStepOptional(activeStep) &&
                  !this.state.completed.has(this.state.activeStep) && (
                    <ButtonWrapper>
                      <Button color="primary" onClick={this.handleSkip}>
                        Skip
                      </Button>
                    </ButtonWrapper>
                  )}
                {activeStep !== steps.length &&
                  (this.state.completed.has(this.state.activeStep) ? (
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

HorizontalNonLinearAlternativeLabelStepper.propTypes = {
  classes: PropTypes.object,
};

export default withStyles(styles)(HorizontalNonLinearAlternativeLabelStepper);
