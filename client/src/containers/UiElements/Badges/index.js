import React, { Component } from 'react';
import { withStyles } from '@material-ui/core/styles';
import LayoutWrapper from '../../../components/utility/layoutWrapper';
import Papersheet, {
  DemoWrapper,
} from '../../../components/utility/papersheet';
import { FullColumn } from '../../../components/utility/rowColumn';
import SimpleBadge from './simpleBadge';

class BadgeExamples extends Component {
  render() {
    const { props } = this;
    return (
      <LayoutWrapper>
        <FullColumn>
          <Papersheet
            title="Badges"
            codeBlock="UiElements/Badges/simpleBadge.js"
          >
            <p>
              Two examples of badges containing text, using primary and accent
              colors. The badge is applied to its children.
            </p>

            <DemoWrapper>
              <SimpleBadge {...props} />
            </DemoWrapper>
          </Papersheet>
        </FullColumn>
      </LayoutWrapper>
    );
  }
}
export default withStyles({})(BadgeExamples);
