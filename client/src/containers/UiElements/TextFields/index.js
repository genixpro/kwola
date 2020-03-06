import React, { Component } from 'react';
import { withStyles } from '@material-ui/core/styles';
import IntlMessages from '../../../components/utility/intlMessages';
import LayoutWrapper from '../../../components/utility/layoutWrapper';
import Papersheet from '../../../components/utility/papersheet';
import { Row, HalfColumn } from '../../../components/utility/rowColumn';

import TextFields from './textFields';
import ComposedTextField from './composedTextFields';
import TextFieldMargins from './layout';
import InputAdornments from './inputAdornments';
import Inputs from './input';
import TextMaskCustom from './formattedInput';
import CustomizedInputs from './customizedInput';

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
              title={<IntlMessages id="sidebar.textfields" />}
              codeBlock="UiElements/TextFields/textFields.js"
              stretched
            >
              <TextFields />
            </Papersheet>
          </HalfColumn>

          <HalfColumn>
            <Papersheet
              title={<IntlMessages id="sidebar.composedTextField" />}
              codeBlock="UiElements/TextFields/composedTextFields.js"
              stretched
            >
              <ComposedTextField />
            </Papersheet>
          </HalfColumn>
        </Row>

        <Row>
          <HalfColumn>
            <Papersheet
              title={<IntlMessages id="sidebar.textFieldMargins" />}
              codeBlock="UiElements/TextFields/layout.js"
              stretched
            >
              <TextFieldMargins />
            </Papersheet>
          </HalfColumn>

          <HalfColumn>
            <Papersheet
              title={<IntlMessages id="sidebar.inputAdornments" />}
              codeBlock="UiElements/TextFields/inputAdornments.js"
              stretched
            >
              <InputAdornments />
            </Papersheet>
          </HalfColumn>
        </Row>

        <Row>
          <HalfColumn>
            <Papersheet
              title={<IntlMessages id="sidebar.inputs" />}
              codeBlock="UiElements/TextFields/input.js"
              stretched
            >
              <Inputs />
            </Papersheet>
          </HalfColumn>

          <HalfColumn>
            <Papersheet
              title={<IntlMessages id="sidebar.textMaskCustom" />}
              codeBlock="UiElements/TextFields/formattedInput.js"
              stretched
            >
              <TextMaskCustom />
            </Papersheet>
          </HalfColumn>
        </Row>

        <Row>
          <HalfColumn>
            <Papersheet
              title={<IntlMessages id="sidebar.customizedInputs" />}
              codeBlock="UiElements/TextFields/customizedInput.js"
            >
              <CustomizedInputs />
            </Papersheet>
          </HalfColumn>
        </Row>
      </LayoutWrapper>
    );
  }
}
export default withStyles(styles, { withTheme: true })(PickerExample);
