import React, { Component } from 'react';
import { withStyles } from '@material-ui/core/styles';
import IntlMessages from '../../../components/utility/intlMessages';
import LayoutWrapper from '../../../components/utility/layoutWrapper';
import Papersheet from '../../../components/utility/papersheet';
import {
  Row,
  HalfColumn,
  FullColumn,
} from '../../../components/utility/rowColumn';
import DefaultButtons from './defaultButton';
import RaisedButtons from './raisedButton';
import ButtonIconLabels from './buttonIconLabel';
import FloatingActionButtons from './floatingActionButton';
import IconButtons from './iconButton';

class ButtonExamples extends Component {
  render() {
    const { props } = this;
    return (
      <LayoutWrapper>
        <Row>
          <HalfColumn>
            <Papersheet
              codeBlock="UiElements/Button/raisedButton.js"
              title={<IntlMessages id="forms.button.raisedButtons" />}
              stretched
            >
              <RaisedButtons {...props} />
            </Papersheet>
          </HalfColumn>

          <HalfColumn>
            <Papersheet
              codeBlock="UiElements/Button/buttonIconLabel.js"
              title={<IntlMessages id="forms.button.buttonIconLabel" />}
              stretched
            >
              <ButtonIconLabels {...props} />
            </Papersheet>
          </HalfColumn>
        </Row>

        <Row>
          <FullColumn>
            <Papersheet
              codeBlock="UiElements/Button/defaultButton.js"
              title={<IntlMessages id="forms.button.defaultButtons" />}
            >
              <DefaultButtons {...props} />
            </Papersheet>
          </FullColumn>
        </Row>

        <Row>
          <HalfColumn>
            <Papersheet
              codeBlock="UiElements/Button/floatingActionButton.js"
              title={<IntlMessages id="forms.button.floatingActionButtons" />}
              stretched
            >
              <FloatingActionButtons {...props} />
            </Papersheet>
          </HalfColumn>

          <HalfColumn>
            <Papersheet
              codeBlock="UiElements/Button/iconButton.js"
              title={<IntlMessages id="forms.button.iconButtons" />}
              stretched
            >
              <IconButtons {...props} />
            </Papersheet>
          </HalfColumn>
        </Row>
      </LayoutWrapper>
    );
  }
}
export default withStyles({})(ButtonExamples);
