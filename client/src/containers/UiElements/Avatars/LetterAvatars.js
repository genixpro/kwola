import React from 'react';
import PropTypes from 'prop-types';
import Avatar from './avatars.style';
import { DemoWrapper } from '../../../components/utility/papersheet';

function LetterAvatars(props) {
  return (
    <DemoWrapper>
      <Avatar>H</Avatar>
      <Avatar className="orangeAvatar">N</Avatar>
      <Avatar className="purpleAvatar">OP</Avatar>
    </DemoWrapper>
  );
}

LetterAvatars.propTypes = {
  classes: PropTypes.object.isRequired,
};

export default LetterAvatars;
