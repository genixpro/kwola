import React, { Component } from 'react';
import { withStyles } from '@material-ui/core/styles';
import LayoutWrapper from '../../../components/utility/layoutWrapper';
import Papersheet, {
  DemoWrapper,
} from '../../../components/utility/papersheet';
import { Row, HalfColumn } from '../../../components/utility/rowColumn';
import SimpleMenu from './simpleMenus';
import SimpleListMenu from './simpleListMenu';
import LongMenu from './longMenu';
import MenuListComposition from './menuListComposition';
import ListItemComposition from './listItemComposition';
import ChangeTransition from './changeTransition';

const styles = theme => ({
  root: {
    width: '100%',
    maxWidth: 360,
    background: theme.palette.background.paper,
    display: 'flex',
  },
  popperClose: {
    pointerEvents: 'none',
  },
  menuItem: {
    '&:focus': {
      backgroundColor: theme.palette.primary.main,
      '& $primary, & $icon': {
        color: theme.palette.common.white,
      },
    },
  },
  primary: {},
  icon: {},
});

class MenuExamples extends Component {
  render() {
    const { props } = this;
    return (
      <LayoutWrapper>
        <Row>
          <HalfColumn>
            <Papersheet
              title="Simple Menu"
              codeBlock="UiElements/Menus/simpleMenus.js"
              stretched
            >
              <p>An example of Simple Menus</p>
              <DemoWrapper>
                <SimpleMenu {...props} />
              </DemoWrapper>
            </Papersheet>
          </HalfColumn>

          <HalfColumn>
            <Papersheet
              title="Simple List Menu"
              codeBlock="UiElements/Menus/simpleListMenu.js"
              stretched
            >
              <p>An example of Simple List Menu</p>
              <DemoWrapper>
                <SimpleListMenu {...props} />
              </DemoWrapper>
            </Papersheet>
          </HalfColumn>
        </Row>
        <Row>
          <HalfColumn>
            <Papersheet
              title="Max height menus"
              codeBlock="UiElements/Menus/longMenu.js"
              stretched
            >
              <p>An example of Max height menus</p>
              <DemoWrapper>
                <LongMenu {...props} />
              </DemoWrapper>
            </Papersheet>
          </HalfColumn>
          <HalfColumn>
            <Papersheet
              title="MenuList composition"
              codeBlock="UiElements/Menus/menuListComposition.js"
              stretched
            >
              <p>An example of MenuList composition</p>
              <DemoWrapper>
                <MenuListComposition {...props} />
              </DemoWrapper>
            </Papersheet>
          </HalfColumn>
        </Row>
        <Row>
          <HalfColumn>
            <Papersheet
              title="List Item composition"
              codeBlock="UiElements/Menus/listItemComposition.js"
              stretched
            >
              <p>An example of List Item composition</p>
              <DemoWrapper>
                <ListItemComposition {...props} />
              </DemoWrapper>
            </Papersheet>
          </HalfColumn>
          <HalfColumn>
            <Papersheet
              title="Change Transition"
              codeBlock="UiElements/Menus/changeTransition.js"
              stretched
            >
              <p>An example of Change Transition</p>
              <DemoWrapper>
                <ChangeTransition {...props} />
              </DemoWrapper>
            </Papersheet>
          </HalfColumn>
        </Row>
      </LayoutWrapper>
    );
  }
}
export default withStyles(styles)(MenuExamples);
