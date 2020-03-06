import React, { Component } from 'react';
import { withStyles } from '@material-ui/core/styles';
import LayoutWrapper from '../../../components/utility/layoutWrapper';
import { Row, HalfColumn } from '../../../components/utility/rowColumn';
import Papersheet, {
  DemoWrapper,
  ContentList,
  BulletListItem,
  Code,
} from '../../../components/utility/papersheet';
import SimpleChips from './SimpleChips';
import ChipsArray from './ChipsArray';

class ChipExamples extends Component {
  render() {
    const { props } = this;
    return (
      <LayoutWrapper>
        <Row>
          <HalfColumn>
            <Papersheet
              title="Chip"
              codeBlock="UiElements/Chips/SimpleChips.js"
              stretched
            >
              <p>
                Examples of Chips, using an image Avatar, SVG Icon Avatar,
                "Letter" and (string) Avatar.
              </p>

              <ContentList>
                <BulletListItem>
                  Chips with the <Code>onClick</Code> property defined change
                  appearance on focus, hover, and click.
                </BulletListItem>
                <BulletListItem>
                  Chips with the <Code>onDelete</Code> property defined will
                  display a delete icon which changes appearance on hover.
                </BulletListItem>
              </ContentList>

              <DemoWrapper>
                <SimpleChips {...props} />
              </DemoWrapper>
            </Papersheet>
          </HalfColumn>

          <HalfColumn>
            <Papersheet
              title="Chip Array"
              codeBlock="UiElements/Chips/ChipsArray.js"
              stretched
            >
              <p>
                An example of rendering multiple Chips from an array of values.
                Deleting a chip removes it from the array. Note that since no{' '}
                <Code>onClick</Code> property is defined, the Chip can be
                focused, but does not gain depth while clicked or touched.
              </p>

              <DemoWrapper>
                <ChipsArray {...props} />
              </DemoWrapper>
            </Papersheet>
          </HalfColumn>
        </Row>
      </LayoutWrapper>
    );
  }
}
export default withStyles({})(ChipExamples);
