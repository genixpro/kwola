import styled from 'styled-components';
import { palette } from 'styled-theme';
import { Fab } from '../../uielements/button';
import Icons from '../../uielements/icon';

const Button = styled(Fab)``;
const Icon = styled(Icons)`
  color: #ffffff;
`;

const MailComposeBtnWrapper = styled.div`
  position: fixed;
  bottom: 25px;
  right: 25px;
  z-index: 1;

  ${Button} {
    background-color: ${palette('red', 7)};

    &:hover {
      background-color: ${palette('red', 9)};
    }

    > span {
      &:last-child {
        span {
          background-color: #ffffff;
        }
      }
    }
  }
`;

export { Button, Icon };
export default MailComposeBtnWrapper;
