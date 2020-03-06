import React, { Component } from 'react';
import { withStyles } from '@material-ui/core/styles';
import IntlMessages from '../../../components/utility/intlMessages';
import LayoutWrapper from '../../../components/utility/layoutWrapper';
import Papersheet from '../../../components/utility/papersheet';
import { Row, HalfColumn } from '../../../components/utility/rowColumn';
import HorizontalNonLinearStepper from './horizontalNonLinear';
import HorizontalLinearStepper from './horizontalLinear';
import HorizontalLabelPositionBelowStepper from './horizontalLinearAlternativeLabel';
import HorizontalNonLinearAlternativeLabelStepper from './horizontalNonLinearAlternativeLabel';
import VerticalLinearStepper from './verticalStepper';
import TextMobileStepper from './mobileStepperText';
import DotsMobileStepper from './mobileStepperDot';
import ProgressMobileStepper from './mobileStepperProgress';

const styles = theme => ({
  card: {
    minWidth: 275,
  },
  bullet: {
    display: 'inline-block',
    margin: '0 2px',
    transform: 'scale(0.8)',
  },
  title: {
    marginBottom: 16,
    fontSize: 14,
    color: theme.palette.text.secondary,
  },
  pos: {
    marginBottom: 12,
    color: theme.palette.text.secondary,
  },

  details: {
    display: 'flex',
    flexDirection: 'column',
  },
  content: {
    flex: '1 0 auto',
  },
  cover: {
    width: 151,
    height: 151,
  },
  controls: {
    display: 'flex',
    alignItems: 'center',
    paddingLeft: theme.spacing(1),
    paddingBottom: theme.spacing(1),
  },
  playIcon: {
    height: 38,
    width: 38,
  },
  media: {
    height: 194,
  },
  expand: {
    transform: 'rotate(0deg)',
    transition: theme.transitions.create('transform', {
      duration: theme.transitions.duration.shortest,
    }),
  },
  expandOpen: {
    transform: 'rotate(180deg)',
  },
  avatar: {
    // backgroundColor: red[500],
  },
  flexGrow: {
    flex: '1 1 auto',
  },
});

class PickerExample extends Component {
  render() {
    return (
      <LayoutWrapper>
        <Row>
          <HalfColumn>
            <Papersheet
              codeBlock="UiElements/Stepper/horizontalLinear.js"
              title={<IntlMessages id="sidebar.horizontalLinearStepper" />}
              stretched
            >
              <HorizontalLinearStepper />
            </Papersheet>
          </HalfColumn>
          <HalfColumn>
            <Papersheet
              codeBlock="UiElements/Stepper/horizontalNonLinear.js"
              title={<IntlMessages id="sidebar.horizontalNonLinearStepper" />}
              stretched
            >
              <HorizontalNonLinearStepper />
            </Papersheet>
          </HalfColumn>
          <HalfColumn>
            <Papersheet
              codeBlock="UiElements/Stepper/horizontalLinearAlternativeLabel.js"
              title={
                <IntlMessages id="sidebar.horizontalLabelPositionBelowStepper" />
              }
              stretched
            >
              <HorizontalLabelPositionBelowStepper />
            </Papersheet>
          </HalfColumn>

          <HalfColumn>
            <Papersheet
              codeBlock="UiElements/Stepper/horizontalNonLinearAlternativeLabel.js"
              title={
                <IntlMessages id="sidebar.horizontalNonLinearAlternativeLabelStepper" />
              }
              stretched
            >
              <HorizontalNonLinearAlternativeLabelStepper />
            </Papersheet>
          </HalfColumn>
          <HalfColumn>
            <Papersheet
              codeBlock="UiElements/Stepper/verticalStepper.js"
              title={<IntlMessages id="sidebar.verticalLinearStepper" />}
              stretched
            >
              <VerticalLinearStepper />
            </Papersheet>
          </HalfColumn>
          <HalfColumn>
            <Papersheet
              codeBlock="UiElements/Stepper/mobileStepperText.js"
              title={<IntlMessages id="sidebar.textMobileStepper" />}
              stretched
            >
              <TextMobileStepper />
            </Papersheet>
          </HalfColumn>
          <HalfColumn>
            <Papersheet
              codeBlock="UiElements/Stepper/mobileStepperDot.js"
              title={<IntlMessages id="sidebar.dotsMobileStepper" />}
            >
              <DotsMobileStepper />
            </Papersheet>
          </HalfColumn>
          <HalfColumn>
            <Papersheet
              codeBlock="UiElements/Stepper/mobileStepperProgress.js"
              title={<IntlMessages id="sidebar.progressMobileStepper" />}
            >
              <ProgressMobileStepper />
            </Papersheet>
          </HalfColumn>
        </Row>
      </LayoutWrapper>
    );
  }
}
export default withStyles(styles, { withTheme: true })(PickerExample);
