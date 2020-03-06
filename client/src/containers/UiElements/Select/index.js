import React, { Component } from 'react';
import { withStyles } from '@material-ui/core/styles';
import IntlMessages from '../../../components/utility/intlMessages';
import LayoutWrapper from '../../../components/utility/layoutWrapper';
import Papersheet from '../../../components/utility/papersheet';
import { Row, HalfColumn } from '../../../components/utility/rowColumn';

import SimpleSelect from './simple';
import NativeSelect from './native';
import MultipleSelect from './multiple';
import DialogSelect from './withDialouge';

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
              codeBlock="UiElements/Select/simple.js"
              title={<IntlMessages id="sidebar.simpleSelect" />}
              stretched
            >
              <SimpleSelect />
            </Papersheet>
          </HalfColumn>

          <HalfColumn>
            <Papersheet
              codeBlock="UiElements/Select/native.js"
              title={<IntlMessages id="sidebar.nativeSelect" />}
              stretched
            >
              <NativeSelect />
            </Papersheet>
          </HalfColumn>
        </Row>
        <Row>
          <HalfColumn>
            <Papersheet
              codeBlock="UiElements/Select/multiple.js"
              title={<IntlMessages id="sidebar.multipleSelect" />}
              stretched
            >
              <MultipleSelect />
            </Papersheet>
          </HalfColumn>

          <HalfColumn>
            <Papersheet
              codeBlock="UiElements/Select/withDialouge.js"
              title={<IntlMessages id="sidebar.withDialougeSelect" />}
              stretched
            >
              <DialogSelect />
            </Papersheet>
          </HalfColumn>
        </Row>
      </LayoutWrapper>
    );
  }
}
export default withStyles(styles, { withTheme: true })(PickerExample);
