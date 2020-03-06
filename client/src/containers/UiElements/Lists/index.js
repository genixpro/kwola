import React, { Component } from 'react';
import { withStyles } from '@material-ui/core/styles';
import LayoutWrapper from '../../../components/utility/layoutWrapper';
import Papersheet, {
  DemoWrapper,
} from '../../../components/utility/papersheet';
import {
  Row,
  HalfColumn,
  FullColumn,
} from '../../../components/utility/rowColumn';
import SimpleList from './simpleList';
import FolderList from './folderList';
import InsetList from './insetList';
import NestedList from './nestedList';
import PinnedSubheaderList from './pinnedSubheaderList';
import CheckboxLists from './checkboxList';
import CheckboxListSecondary from './checkboxListSecondary';
import SwitchListSecondary from './switchListSecondary';
import InteractiveList from './interactiveList';

const styles = theme => ({
  root: {
    width: '100%',
    background: theme.palette.background.paper,
    position: 'relative',
    overflow: 'auto',
    maxHeight: 300,
    flexGrow: 1,
    maxWidth: 752,
  },
  demo: {
    background: theme.palette.background.paper,
  },
  title: {
    margin: `${theme.spacing(4)}px 0 ${theme.spacing(2)}px`,
  },
  nested: {
    paddingLeft: theme.spacing(4),
  },
  listSection: {
    background: 'inherit',
  },
});

class ListExamples extends Component {
  render() {
    const { props } = this;
    return (
      <LayoutWrapper>
        <Row>
          <HalfColumn>
            <Papersheet
              title="Simple List"
              codeBlock="UiElements/Lists/simpleList.js"
              stretched
            >
              <DemoWrapper>
                <SimpleList {...props} />
              </DemoWrapper>
            </Papersheet>
          </HalfColumn>

          <HalfColumn>
            <Papersheet
              title="Folder List"
              codeBlock="UiElements/Lists/folderList.js"
              stretched
            >
              <DemoWrapper>
                <FolderList {...props} />
              </DemoWrapper>
            </Papersheet>
          </HalfColumn>
        </Row>
        <Row>
          <HalfColumn>
            <Papersheet
              title="Inset List"
              codeBlock="UiElements/Lists/insetList.js"
              stretched
            >
              <DemoWrapper>
                <InsetList {...props} />
              </DemoWrapper>
            </Papersheet>
          </HalfColumn>

          <HalfColumn>
            <Papersheet
              title="Nested List"
              codeBlock="UiElements/Lists/nestedList.js"
              stretched
            >
              <DemoWrapper>
                <NestedList {...props} />
              </DemoWrapper>
            </Papersheet>
          </HalfColumn>
        </Row>
        <Row>
          <HalfColumn>
            <Papersheet
              title="Pinned Subheader List"
              codeBlock="UiElements/Lists/pinnedSubheaderList.js"
              stretched
            >
              <DemoWrapper>
                <PinnedSubheaderList {...props} />
              </DemoWrapper>
            </Papersheet>
          </HalfColumn>

          <HalfColumn>
            <Papersheet
              title="Checkbox List"
              codeBlock="UiElements/Lists/checkboxLists.js"
              stretched
            >
              <DemoWrapper>
                <CheckboxLists {...props} />
              </DemoWrapper>
            </Papersheet>
          </HalfColumn>
        </Row>
        <Row>
          <HalfColumn>
            <Papersheet
              title="Secondary Checkbox List"
              codeBlock="UiElements/Lists/checkboxListSecondary.js"
              stretched
            >
              <DemoWrapper>
                <CheckboxListSecondary {...props} />
              </DemoWrapper>
            </Papersheet>
          </HalfColumn>

          <HalfColumn>
            <Papersheet
              title="Switch Checkbox List"
              codeBlock="UiElements/Lists/switchListSecondary.js"
              stretched
            >
              <DemoWrapper>
                <SwitchListSecondary {...props} />
              </DemoWrapper>
            </Papersheet>
          </HalfColumn>
        </Row>
        <Row>
          <FullColumn>
            <Papersheet
              title="Interactive List"
              codeBlock="UiElements/Lists/interactiveList.js"
            >
              <InteractiveList {...props} />
            </Papersheet>
          </FullColumn>
        </Row>
      </LayoutWrapper>
    );
  }
}
export default withStyles(styles)(ListExamples);
