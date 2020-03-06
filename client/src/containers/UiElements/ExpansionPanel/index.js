import React, { Component } from 'react';
import { withStyles } from '@material-ui/core/styles';
import SimplePanel from './simplePanel';
import SecondaryPanel from './secondaryPanel';
import ControlledAccordion from './controlledAccordion';
import LayoutWrapper from '../../../components/utility/layoutWrapper';
import Papersheet, {
  DemoWrapper,
} from '../../../components/utility/papersheet';
import { Row, FullColumn } from '../../../components/utility/rowColumn';

const styles = theme => ({
  root: {
    width: '100%',
  },
  heading: {
    fontSize: theme.typography.pxToRem(15),
    fontWeight: theme.typography.fontWeightRegular,
    flexBasis: '33.33%',
    flexShrink: 0,
  },
  secondaryHeading: {
    fontSize: theme.typography.pxToRem(15),
    color: theme.palette.text.secondary,
  },
  icon: {
    verticalAlign: 'bottom',
    height: 20,
    width: 20,
  },
  details: {
    alignItems: 'center',
  },
  column: {
    flexBasis: '33.3%',
  },
  helper: {
    borderLeft: `2px solid ${theme.palette.text.lightDivider}`,
    padding: `${theme.spacing(1)}px ${theme.spacing(2)}px`,
  },
  link: {
    color: theme.palette.primary[500],
    textDecoration: 'none',
    '&:hover': {
      textDecoration: 'underline',
    },
  },
});

class ExpansionPanelExamples extends Component {
  render() {
    const { props } = this;
    return (
      <LayoutWrapper>
        <Row>
          <FullColumn>
            <Papersheet
              title="Simple Expansion Panel"
              codeBlock="UiElements/ExpansionPanel/simplePanel.js"
            >
              <p>An example of Simple Expansion Panel</p>
              <DemoWrapper>
                <SimplePanel {...props} />
              </DemoWrapper>
            </Papersheet>
          </FullColumn>
        </Row>
        <Row>
          <FullColumn>
            <Papersheet
              title="Secondary Expansion Panel"
              codeBlock="UiElements/ExpansionPanel/secondaryPanel.js"
            >
              <p>An example of secondary Expansion Panel</p>
              <DemoWrapper>
                <SecondaryPanel {...props} />
              </DemoWrapper>
            </Papersheet>
          </FullColumn>
        </Row>
        <Row>
          <FullColumn>
            <Papersheet
              title="Controlled Accordion Panel"
              codeBlock="UiElements/ExpansionPanel/controlledAccordion.js"
            >
              <p>An example of Controlled Accordion Panel</p>
              <DemoWrapper>
                <ControlledAccordion {...props} />
              </DemoWrapper>
            </Papersheet>
          </FullColumn>
        </Row>
      </LayoutWrapper>
    );
  }
}
export default withStyles(styles)(ExpansionPanelExamples);
