import React, { Component } from 'react';
import { withStyles } from '@material-ui/core/styles';
import LayoutWrapper from '../../../components/utility/layoutWrapper';
import { Row, HalfColumn } from '../../../components/utility/rowColumn';
import Papersheet from '../../../components/utility/papersheet';
import ListDivider from './ListDivider';
import InsetDivider from './InsetDivider';

class DividerExamples extends Component {
  render() {
    const { props } = this;
    return (
      <LayoutWrapper>
        <Row>
          <HalfColumn>
            <Papersheet
              title="List Divider"
              codeBlock="UiElements/Dividers/ListDivider.js"
              stretched
            >
              <ListDivider {...props} />
            </Papersheet>
          </HalfColumn>

          <HalfColumn>
            <Papersheet
              title="Inset Divider"
              codeBlock="UiElements/Dividers/InsetDivider.js"
              stretched
            >
              <InsetDivider {...props} />
            </Papersheet>
          </HalfColumn>
        </Row>
      </LayoutWrapper>
    );
  }
}
export default withStyles({})(DividerExamples);
