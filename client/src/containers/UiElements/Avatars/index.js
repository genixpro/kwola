import React, { Component } from 'react';
import { withStyles } from '@material-ui/core/styles';
import LayoutWrapper from '../../../components/utility/layoutWrapper';
import { Row, HalfColumn } from '../../../components/utility/rowColumn';
import Papersheet, { Code } from '../../../components/utility/papersheet';
import pink from '@material-ui/core/colors/pink';
import green from '@material-ui/core/colors/green';
import IconAvatars from './IconAvatars';
import ImageAvatars from './ImageAvatars';
import LetterAvatars from './LetterAvatars';
const styles = {
  row: {
    display: 'flex',
    justifyContent: 'center',
  },
  avatar: {
    margin: 10,
  },
  bigAvatar: {
    width: 60,
    height: 60,
  },
  pinkAvatar: {
    margin: 10,
    color: '#fff',
    backgroundColor: pink[500],
  },
  greenAvatar: {
    margin: 10,
    color: '#fff',
    backgroundColor: green[500],
  },
};

class AvatarExamples extends Component {
  render() {
    const { props } = this;
    return (
      <LayoutWrapper>
        <Row>
          <HalfColumn>
            <Papersheet
              title="Icon Avatars"
              codeBlock="UiElements/Avatars/IconAvatars.js"
              stretched
            >
              <p>
                Image avatars can be created by passing standard{' '}
                <Code>img</Code> props <Code>src</Code> or <Code>srcSet</Code>{' '}
                into the component.
              </p>
              <IconAvatars {...props} />
            </Papersheet>
          </HalfColumn>
          <HalfColumn>
            <Papersheet
              title="Image Avatars"
              codeBlock="UiElements/Avatars/LetterAvatars.js"
              stretched
            >
              <p>
                Icon avatars are created by passing an icon as{' '}
                <Code>children</Code>.
              </p>
              <ImageAvatars {...props} />
            </Papersheet>
          </HalfColumn>
          <HalfColumn>
            <Papersheet
              title="Letter Avatars"
              codeBlock="UiElements/Avatars/ImageAvatars.js"
            >
              <p>
                Avatars containing simple characters can be created by passing
                your string as <Code>children</Code>.
              </p>
              <LetterAvatars {...props} />
            </Papersheet>
          </HalfColumn>
        </Row>
      </LayoutWrapper>
    );
  }
}
export default withStyles(styles)(AvatarExamples);
