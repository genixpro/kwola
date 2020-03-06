import React, { Component } from 'react';
import { withStyles } from '@material-ui/core/styles';
import LayoutWrapper from '../../../components/utility/layoutWrapper';
import Papersheet, {
  DemoWrapper
} from '../../../components/utility/papersheet';
import { Row,FullColumn } from '../../../components/utility/rowColumn';
import SimpleModal from './simpleModal';

const styles = {
  root: {
    width: 500,
  },
};


class SimpleModalExample extends Component {
  render() {
    const { props } = this;
    return (
      <LayoutWrapper>
          <Row>
            <FullColumn>
              <Papersheet
                title="Simple Modal"
                codeBlock="UiElements/Modals/simpleModal.js"
              >
              <p>
                An Example of Simple Modal
              </p>
              <DemoWrapper>
                <SimpleModal {...props} />
              </DemoWrapper>
              </Papersheet>
            </FullColumn>
            </Row>
      </LayoutWrapper>
    );
  }
}
export default withStyles(styles)(SimpleModalExample);
