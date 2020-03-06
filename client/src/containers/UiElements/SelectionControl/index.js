import React, { Component } from 'react';
import { withStyles } from '@material-ui/core/styles';
import IntlMessages from '../../../components/utility/intlMessages';
import LayoutWrapper from '../../../components/utility/layoutWrapper';
import Papersheet from '../../../components/utility/papersheet';
import { Row, HalfColumn } from '../../../components/utility/rowColumn';

import Checkbox from './checkbox/checkbox';
import CheckboxGroup from './checkbox/checkboxGroup';
import RadioButtonsGroup from './radio/radioButton';
import RadioButtons from './radio/radioWithoutWrapper';
import Switches from './switch/switch';
import SwitchLabels from './switch/switchWithLabel';

class PickerExample extends Component {
  render() {
    return (
      <LayoutWrapper>
        <Row>
          <HalfColumn>
            <Papersheet
              codeBlock="UiElements/SelectionControl/checkbox/checkbox.js"
              title={<IntlMessages id="sidebar.checkbox" />}
              stretched
            >
              <Checkbox />
            </Papersheet>
          </HalfColumn>

          <HalfColumn>
            <Papersheet
              codeBlock="UiElements/SelectionControl/checkbox/checkboxGroup.js"
              title="Checkbox Group"
              stretched
            >
              <CheckboxGroup />
            </Papersheet>
          </HalfColumn>
        </Row>
        <Row>
          <HalfColumn>
            <Papersheet
              codeBlock="UiElements/SelectionControl/radio/radioButton.js"
              title="Raido Buttons Group"
              stretched
            >
              <RadioButtonsGroup />
            </Papersheet>
          </HalfColumn>

          <HalfColumn>
            <Papersheet
              codeBlock="UiElements/SelectionControl/radio/radioWithoutWrapper.js"
              title="Radio Buttons"
              stretched
            >
              <RadioButtons />
            </Papersheet>
          </HalfColumn>
        </Row>
        <Row>
          <HalfColumn>
            <Papersheet
              codeBlock="UiElements/SelectionControl/switch/switch.js"
              title={<IntlMessages id="sidebar.switch" />}
              stretched
            >
              <Switches />
            </Papersheet>
          </HalfColumn>

          <HalfColumn>
            <Papersheet
              codeBlock="UiElements/SelectionControl/switch/switchWithLabel.js"
              title="Switch Labels"
              stretched
            >
              <SwitchLabels />
            </Papersheet>
          </HalfColumn>
        </Row>
      </LayoutWrapper>
    );
  }
}
export default withStyles({})(PickerExample);
