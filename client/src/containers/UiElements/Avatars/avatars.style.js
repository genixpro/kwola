import styled from 'styled-components';
import { palette } from 'styled-theme';
import Avatars from '../../../components/uielements/avatars';

const Avatar = styled(Avatars)`
  margin: 10px;

  &.orangeAvatar {
    color: #fffffff;
    background-color: ${palette('orange', 5)};
  }

  &.purpleAvatar {
    color: #fffffff;
    background-color: ${palette('purple', 5)};
  }

  &.pinkAvatar {
    color: #fffffff;
    background-color: ${palette('pink', 5)};
  }

  &.greenAvatar {
    color: #fffffff;
    background-color: ${palette('green', 5)};
  }

  &.bigAvatar {
    width: 60px;
    height: 60px;
  }
`;

export default Avatar;
