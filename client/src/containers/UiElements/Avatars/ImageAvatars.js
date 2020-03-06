import React from 'react';
import PropTypes from 'prop-types';
import Avatar from './avatars.style';

import AvatarPic1 from '../../../images/avatars/imageavatar-1.jpg';
import AvatarPic2 from '../../../images/avatars/imageavatar-2.jpg';
import { DemoWrapper } from '../../../components/utility/papersheet';

function ImageAvatars(props) {
  return (
    <DemoWrapper>
      <Avatar alt="Remy Sharp" src={AvatarPic1} />
      <Avatar alt="Adelle Charles" src={AvatarPic2} className="bigAvatar" />
    </DemoWrapper>
  );
}

ImageAvatars.propTypes = {
  classes: PropTypes.object.isRequired,
};

export default ImageAvatars;
